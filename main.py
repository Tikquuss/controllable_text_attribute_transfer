# coding: utf-8
import time
import argparse
import os
import torch
import torch.nn as nn
import copy
import tqdm

from model import make_model, make_deb, Classifier, LabelSmoothing, fgim_attack, fgim, LossSedat
from data import prepare_data, non_pair_data_loader, get_cuda, id2text_sentence, to_var, load_human_answer
from utils import bool_flag, add_log, add_output, write_text_z_in_file
from optim import get_optimizer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def preparation(args):
    # set model save path
    if args.if_load_from_checkpoint:
        timestamp = args.checkpoint_name
    else:
        timestamp = str(int(time.time()))
        print("create new model save path: %s" % timestamp)
    args.current_save_path = os.path.join(args.dump_path, timestamp) # 'save/%s/' % timestamp
    args.log_file = os.path.join(args.current_save_path, time.strftime("log_%Y_%m_%d_%H_%M_%S.txt", time.localtime()))
    args.output_file = os.path.join(args.current_save_path, time.strftime("output_%Y_%m_%d_%H_%M_%S.txt", time.localtime()))
    print("create log file at path: %s" % args.log_file)

    if os.path.exists(args.current_save_path):
        add_log(args, "Load checkpoint model from Path: %s" % args.current_save_path)
    else:
        os.makedirs(args.current_save_path)
        add_log(args, "Path: %s is created" % args.current_save_path)

    # prepare data
    args.id_to_word, args.vocab_size = prepare_data(args)

def train_iters(args, ae_model, dis_model):
    train_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    file_list = [args.train_data_file]
    if os.path.exists(args.val_data_file) :
        file_list.append(args.val_data_file)

    train_data_loader.create_batches(args, file_list, if_shuffle=True)
    add_log(args, "Start train process.")
    ae_model.train()
    dis_model.train()

    ae_optimizer = get_optimizer(parameters=ae_model.parameters(), s=args.ae_optimizer, noamopt=args.ae_noamopt)
    dis_optimizer = get_optimizer(parameters=dis_model.parameters(), s=args.dis_optimizer)
    
    ae_criterion = get_cuda(LabelSmoothing(size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1))
    dis_criterion = nn.BCELoss(size_average=True)

    stats = {}
    for epoch in range(args.max_epochs):
        print('-' * 94)
        epoch_start_time = time.time()
        loss_ae, n_words_ae, xe_loss_ae, n_valid_ae = 0, 0, 0, 0
        for it in range(train_data_loader.num_batch):
            flag_rec = True
            batch_sentences, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, tensor_ntokens = train_data_loader.next_batch()

            # Forward pass
            latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)

            # Loss calculation
            if not args.sedat :
                loss_rec = ae_criterion(out.contiguous().view(-1, out.size(-1)),
                                        tensor_tgt_y.contiguous().view(-1)) / tensor_ntokens.data

            else :
                # only on positive example
                positive_examples = tensor_labels.squeeze()==args.positive_label
                out = out[positive_examples] # or out[positive_examples,:,:]
                tensor_tgt_y = tensor_tgt_y[positive_examples] # or tensor_tgt_y[positive_examples,:] 
                tensor_ntokens = (tensor_tgt_y != 0).data.sum().float() 
                loss_rec = ae_criterion(out.contiguous().view(-1, out.size(-1)),
                                        tensor_tgt_y.contiguous().view(-1)) / tensor_ntokens.data
                flag_rec = positive_examples.any()
            
            #tensor_tgt_y : seq_len, bs
            #n_words = tensor_tgt_y.size(0)
            #xe_loss = loss_rec.item() * len(tensor_tgt_y)
            #n_valid = (out.round().int() == tensor_tgt_y).sum().item()
            #stats_["mlm_loss"] = loss.item()
            #stats_["mlm_ppl"] = np.exp(xe_loss / n_words)
            #stats_["mlm_acc"] = 100. * n_valid / n_words
                        
            ae_optimizer.zero_grad()

            loss_rec.backward()
            ae_optimizer.step()

            # Classifier
            dis_lop = dis_model.forward(to_var(latent.clone()))

            loss_dis = dis_criterion(dis_lop, tensor_labels)

            dis_optimizer.zero_grad()
            loss_dis.backward()
            dis_optimizer.step()

            if it % args.log_interval == 0:
                add_log(args, 
                    '| epoch {:3d} | {:5d}/{:5d} batches | rec loss {:5.4f} | dis loss {:5.4f} |'.format(
                        epoch, it, train_data_loader.num_batch, loss_rec, loss_dis))
                if flag_rec :
                    print(id2text_sentence(tensor_tgt_y[0], args.id_to_word))
                    generator_text = ae_model.greedy_decode(latent,
                                                            max_len=args.max_sequence_length,
                                                            start_id=args.id_bos)
                    # batch_sentences
                    print(id2text_sentence(generator_text[0], args.id_to_word))

        add_log(args, 
            '| end of epoch {:3d} | time: {:5.2f}s |'.format(
                epoch, (time.time() - epoch_start_time)))
        
        # Save model
        torch.save(ae_model.state_dict(), os.path.join(args.current_save_path, 'ae_model_params.pkl'))
        torch.save(dis_model.state_dict(), os.path.join(args.current_save_path, 'dis_model_params.pkl'))

def eval_iters(args, ae_model, dis_model):
    batch_size=1
    eval_data_loader = non_pair_data_loader(
        batch_size=batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    file_list = [args.test_data_file]
    eval_data_loader.create_batches(args, file_list, if_shuffle=False)
    if args.references_files :
        gold_ans = load_human_answer(args.references_files, args.text_column)
        assert len(gold_ans) == eval_data_loader.num_batch
    else :
        gold_ans = [[None]*batch_size]*eval_data_loader.num_batch

    add_log(args, "Start eval process.")
    ae_model.eval()
    dis_model.eval()

    fgim_our = True
    if fgim_our :
        # for FGIM
        z_prime, text_z_prime = fgim(eval_data_loader, args, ae_model, dis_model, gold_ans = gold_ans)
        write_text_z_in_file(args, text_z_prime)
        
    else :
        for it in range(eval_data_loader.num_batch):
            batch_sentences, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, tensor_ntokens = eval_data_loader.next_batch()

            print("------------%d------------" % it)
            print(id2text_sentence(tensor_tgt_y[0], args.id_to_word))
            print("origin_labels", tensor_labels)

            latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)
            generator_text = ae_model.greedy_decode(latent,
                                                    max_len=args.max_sequence_length,
                                                    start_id=args.id_bos)
            print(id2text_sentence(generator_text[0], args.id_to_word))

            # Define target label
            target = get_cuda(torch.tensor([[1.0]], dtype=torch.float))
            if tensor_labels[0].item() > 0.5:
                target = get_cuda(torch.tensor([[0.0]], dtype=torch.float))
            print("target_labels", target)

            modify_text = fgim_attack(dis_model, latent, target, ae_model, args.max_sequence_length, args.id_bos,
                                            id2text_sentence, args.id_to_word, gold_ans[it])
                    
            add_output(args, modify_text)
            

def sedat_train(args, ae_model, f, deb) :
    """
    Input: 
        Original latent representation z : (n_batch, batch_size, seq_length, latent_size)
    Output: 
        An optimal modified latent representation z'
    """
    lambda_ = 0.5
    alpha, beta = 1., 1.
    
    train_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    file_list = [args.train_data_file]
    if os.path.exists(args.val_data_file) :
        file_list.append(args.val_data_file)
    train_data_loader.create_batches(args, file_list, if_shuffle=True)
    
    add_log(args, "Start train process.")
            
    #add_log("Start train process.")
    ae_model.train()
    f.train()
    deb.train()
    
    ae_optimizer = get_optimizer(parameters=ae_model.parameters(), s=args.ae_optimizer, noamopt=args.ae_noamopt)
    dis_optimizer = get_optimizer(parameters=f.parameters(), s=args.dis_optimizer)
    deb_optimizer = get_optimizer(parameters=deb.parameters(), s=args.dis_optimizer)
    
    ae_criterion = get_cuda(LabelSmoothing(size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1))
    dis_criterion = nn.BCELoss(size_average=True)
    deb_criterion = LossSedat(penalty="lasso")

    for epoch in range(args.max_epochs):
        print('-' * 94)
        epoch_start_time = time.time()
        
        for it in range(train_data_loader.num_batch):
            _, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, _ = train_data_loader.next_batch()
            only_on_negative_example = False
            # only on negative example
            flag = True
            if only_on_negative_example :
                negative_examples = ~(tensor_labels.squeeze()==args.positive_label)
                tensor_labels = tensor_labels[negative_examples]
                tensor_src = tensor_src[negative_examples]
                tensor_src_mask = tensor_src_mask[negative_examples] 
                tensor_tgt_y = tensor_tgt_y[negative_examples]  
                tensor_tgt = tensor_tgt[negative_examples]
                tensor_tgt_mask = tensor_tgt_mask[negative_examples]
                flag = negative_examples.any()
            if flag :
                # forward
                z, out, z_list = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask, return_intermediate=True)
                #y_hat = f.forward(to_var(z.clone()))
                y_hat = f.forward(z)
                
                loss_dis = dis_criterion(y_hat, tensor_labels)
                dis_optimizer.zero_grad()
                loss_dis.backward(retain_graph=True)
                dis_optimizer.step()

                mask_deb = y_hat.squeeze()>=lambda_ if args.positive_label==0 else y_hat.squeeze()<lambda_
                # if f(z) > lambda :
                if mask_deb.any() :
                    y_hat_deb = y_hat[mask_deb]
                    z_deb = z[mask_deb]
                    z_prime, z_prime_list = deb(z_deb, mask=None, return_intermediate=True)
                    #z_prime = z_prime[mask_deb]
                    z_prime = torch.sum(ae_model.sigmoid(z_prime), dim=1)
                    if True :
                        loss_deb = alpha * deb_criterion(z_deb, z_prime, is_list = False) + beta * y_hat_deb.sum()
                    else :
                        # TODO : shape problem, torch.Size([32, 59, 256]) vs torch.Size([32, 32, 256])
                        z_deb_list = [z_[mask_deb] for z_ in z_list]
                        loss_deb = alpha * deb_criterion(z_deb_list, z_prime_list, is_list = True) + beta * y_hat_deb.sum()
                
                    deb_optimizer.zero_grad()
                    loss_deb.backward(retain_graph=True)
                    deb_optimizer.step()
                else :
                    loss_deb = float("nan")
                
                # else :
                if (~mask_deb).any() :
                    out_ = out[~mask_deb] 
                    tensor_tgt_y_ = tensor_tgt_y[~mask_deb] 
                    tensor_ntokens = (tensor_tgt_y_ != 0).data.sum().float() 
                    loss_rec = ae_criterion(out_.contiguous().view(-1, out_.size(-1)),
                                                    tensor_tgt_y_.contiguous().view(-1)) / tensor_ntokens.data
                else :
                    loss_rec = float("nan")
                    
                ae_optimizer.zero_grad()
                (loss_dis + loss_deb + loss_rec).backward()
                ae_optimizer.step()
                
                if it % args.log_interval == 0:
                    add_log(args, 
                        '| epoch {:3d} | {:5d}/{:5d} batches | rec loss {:5.4f} | dis loss {:5.4f} | deb loss {:5.4f} |'.format(
                            epoch, it, train_data_loader.num_batch, loss_rec, loss_dis, loss_deb))
                    
                    print("input : ", id2text_sentence(tensor_tgt_y[0], args.id_to_word))
                    generator_text = ae_model.greedy_decode(z,
                                                            max_len=args.max_sequence_length,
                                                            start_id=args.id_bos)
                    # batch_sentences
                    print("gen : ", id2text_sentence(generator_text[0], args.id_to_word))
                    if mask_deb.any() :
                        generator_text_prime = ae_model.greedy_decode(z_prime,
                                                                max_len=args.max_sequence_length,
                                                                start_id=args.id_bos)

                        print("deb : ",id2text_sentence(generator_text_prime[0], args.id_to_word))

        add_log(args, 
            '| end of epoch {:3d} | time: {:5.2f}s |'.format(
                epoch, (time.time() - epoch_start_time)))
        
        # Save model
        torch.save(ae_model.state_dict(), os.path.join(args.current_save_path, 'ae_model_params_deb.pkl'))
        torch.save(f.state_dict(), os.path.join(args.current_save_path, 'dis_model_params_deb.pkl'))
        torch.save(deb.state_dict(), os.path.join(args.current_save_path, 'deb_model_params_deb.pkl'))

        
def sedat_eval(args, ae_model, f, deb) :
    """
    Input: 
        Original latent representation z : (n_batch, batch_size, seq_length, latent_size)
    Output: 
        An optimal modified latent representation z'
    """
    max_sequence_length = args.max_sequence_length
    id_bos = args.id_bos
    id_to_word = args.id_to_word
    limit_batches = args.limit_batches

    eval_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    file_list = [args.test_data_file]
    eval_data_loader.create_batches(args, file_list, if_shuffle=False)
    if args.references_files :
        gold_ans = load_human_answer(args.references_files, args.text_column)
        assert len(gold_ans) == eval_data_loader.num_batch
    else :
        gold_ans = None

    add_log(args, "Start eval process.")
    ae_model.eval()
    f.eval()
    deb.eval()
    
    text_z_prime = {}
    text_z_prime = {"source" : [], "origin_labels" : [], "before" : [], "after" : [], "change" : [], "pred_label" : []}
    if gold_ans is not None :
        text_z_prime["gold_ans"] = []
    z_prime = []
    n_batches = 0
    for it in tqdm.tqdm(list(range(eval_data_loader.num_batch)), desc="FGIM"):
        
        _, tensor_labels, \
        tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
        tensor_tgt_mask, _ = eval_data_loader.next_batch()
        # only on negative example
        negative_examples = ~(tensor_labels.squeeze()==args.positive_label)
        tensor_labels = tensor_labels[negative_examples]
        tensor_src = tensor_src[negative_examples]
        tensor_src_mask = tensor_src_mask[negative_examples] 
        tensor_tgt_y = tensor_tgt_y[negative_examples]  
        tensor_tgt = tensor_tgt[negative_examples]
        tensor_tgt_mask = tensor_tgt_mask[negative_examples]
        if negative_examples.any():
            if gold_ans is not None :
                text_z_prime["gold_ans"].append(gold_ans[it])
            
            text_z_prime["source"].append([id2text_sentence(t, args.id_to_word) for t in tensor_tgt_y])
            text_z_prime["origin_labels"].append(tensor_labels.cpu().numpy())
            
            origin_data, _ = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)
            
            generator_id = ae_model.greedy_decode(origin_data, max_len=max_sequence_length, start_id=id_bos)
            generator_text = [id2text_sentence(gid, id_to_word) for gid in generator_id]
            text_z_prime["before"].append(generator_text)
            
            data = deb(origin_data)
            data = torch.sum(ae_model.sigmoid(data), dim=1)  # (batch_size, d_model)
            logit = ae_model.decode(data.unsqueeze(1), tensor_tgt, tensor_tgt_mask)  # (batch_size, max_tgt_seq, d_model)
            output = ae_model.generator(logit)  # (batch_size, max_seq, vocab_size)      
            z_prime.append(data)
            generator_id = ae_model.greedy_decode(data, max_len=max_sequence_length, start_id=id_bos)
            generator_text = [id2text_sentence(gid, id_to_word) for gid in generator_id]
            text_z_prime["after"].append(generator_text)
            text_z_prime["change"].append([True]*len(output))
            text_z_prime["pred_label"].append([o.item() for o in output])
            
            n_batches += 1
            if n_batches > limit_batches:
                break 
    write_text_z_in_file(args, text_z_prime)       
    return z_prime, text_z_prime
    
    
def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Here is your model discription.")

    # main parameters
    ######################################################################################
    #  Environmental parameters
    ######################################################################################
    parser.add_argument('--id_pad', type=int, default=0, help='')
    parser.add_argument('--id_unk', type=int, default=1, help='')
    parser.add_argument('--id_bos', type=int, default=2, help='')
    parser.add_argument('--id_eos', type=int, default=3, help='')

    ######################################################################################
    #  File parameters
    ######################################################################################
    parser.add_argument("--dump_path", type=str, default="save", help="Experiment dump path")
    parser.add_argument('--data_path', type=str, default='', help='')
    parser.add_argument('--train_data_file', type=str, default='', help='')
    parser.add_argument('--val_data_file', type=str, default='', help='')
    parser.add_argument('--test_data_file', type=str, default='', help='')
    parser.add_argument('--references_files', type=str, default='', help='')
    parser.add_argument('--word_to_id_file', type=str, default='', help='')
    parser.add_argument("-dc", "--data_columns", type=str, default="c1,c2,..", help="")

    ######################################################################################
    #  Model parameters
    ######################################################################################
    parser.add_argument('--word_dict_max_num', type=int, default=5, help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--num_layers_AE', type=int, default=2)
    parser.add_argument('--transformer_model_size', type=int, default=256)
    parser.add_argument('--transformer_ff_size', type=int, default=1024)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--word_dropout', type=float, default=1.0)
    parser.add_argument('--embedding_dropout', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--label_size', type=int, default=1)
    
    ######################################################################################
    # Training
    ###################################################################################### 
    parser.add_argument('--max_epochs', type=int, default=10) 
    parser.add_argument('--log_interval', type=int, default=100)  
    parser.add_argument('--eval_only', type=bool_flag, default=False) 
    parser.add_argument('--sedat', type=bool_flag, default=False)
    parser.add_argument('--positive_label', type=int, default=0) 
    
    parser.add_argument('--w', type=str, default="2.0,3.0,4.0,5.0,6.0,7.0,8.0")
    parser.add_argument('--lambda_', type=float, default=0.9)
    parser.add_argument('--threshold', type=float, default=0.001)
    parser.add_argument('--max_iter_per_epsilon', type=int, default=100) 
    parser.add_argument('--limit_batches', type=int, default=-1)
    
    parser.add_argument('--task', type = str, default="pretrain", choices=["pretrain", "debias"], help=  "")
    
    ######################################################################################
    # Optimizer
    ######################################################################################     
    parser.add_argument('--ae_noamopt', type=str, default="factor_ae=1,warmup_ae=200")
    parser.add_argument("--ae_optimizer", type=str, default="adam,lr=0,beta1=0.9,beta2=0.98,eps=0.000000001")
    parser.add_argument("--dis_optimizer", type=str, default="adam,lr=0.0001")
    parser.add_argument("--deb_optimizer", type=str, default="adam,lr=0.0001")

    ######################################################################################
    # Checkpoint
    ######################################################################################
    parser.add_argument('--if_load_from_checkpoint', type=bool_flag, default=False)
    #if parser.parse_known_args()[0].if_load_from_checkpoint:
    parser.add_argument('--checkpoint_name', type=str, default='', help='1557667911, ...')

    ######################################################################################
    #  End of hyper parameters
    ######################################################################################
    
    return parser

def main(args):
    preparation(args)
    
    ae_model = get_cuda(make_model(d_vocab=args.vocab_size,
                                    N=args.num_layers_AE,
                                    d_model=args.transformer_model_size,
                                    latent_size=args.latent_size,
                                    d_ff=args.transformer_ff_size,
                                    h=args.n_heads, 
                                    dropout=args.attention_dropout
    ))
    dis_model = get_cuda(Classifier(latent_size=args.latent_size, output_size=args.label_size))
    
    if args.task == "debias" :
        load_db_from_ae_model = False
        if load_db_from_ae_model :
            deb_model = copy.deepcopy(ae_model.encoder)
        else :
            deb_model= get_cuda(make_deb(N=args.num_layers_AE, 
                                d_model=args.transformer_model_size, 
                                d_ff=args.transformer_ff_size, h=args.n_heads, 
                                dropout=args.attention_dropout))

    if args.if_load_from_checkpoint:
        # Load models' params from checkpoint
        ae_model.load_state_dict(torch.load(os.path.join(args.current_save_path, 'ae_model_params.pkl')))
        dis_model.load_state_dict(torch.load(os.path.join(args.current_save_path, 'dis_model_params.pkl')))
        f1 = os.path.join(args.current_save_path, 'ae_model_params_deb.pkl')
        f2 = os.path.join(args.current_save_path, 'dis_model_params_deb.pkl')
        if os.path.exists(f1):
            ae_model.load_state_dict(torch.load(f1))
            dis_model.load_state_dict(torch.load(f2))
        f3 = os.path.join(args.current_save_path, 'deb_model_params_deb.pkl')
        if args.task == "debias" and os.path.exists(f1) :
            deb_model.load_state_dict(torch.load(f3))
        
    if not args.eval_only:
        if args.task == "pretrain":
            train_iters(args, ae_model, dis_model)
        if args.task == "debias" :
            sedat_train(args, ae_model, f=dis_model, deb=deb_model) 
            
    if os.path.exists(args.test_data_file) :
        if args.task == "pretrain":
            eval_iters(args, ae_model, dis_model)
        if args.task == "debias" :
            sedat_eval(args, ae_model, f=dis_model, deb=deb_model) 

    print("Done!")

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    args = parser.parse_args()

    # check parameters
    #assert os.path.exists(args.data_path)
    data_columns = args.data_columns
    data_columns = data_columns.split(",")
    assert len(data_columns) == 2
    args.text_column = data_columns[0]
    args.label_column = data_columns[1]
    
    references_files = args.references_files.strip().strip('"')
    if references_files != "" :
        references_files = references_files.split(",")
        #assert all([os.path.isfile(f) or f==""  for f in references_files])
        #args.references_files = references_files
        args.references_files = []
        for f in references_files :
            assert os.path.isfile(f) or f==""
            if f != "":
                args.references_files.append(f)
    else :
        args.references_files = []

    if args.eval_only:
        assert os.path.exists(args.test_data_file)
        assert args.if_load_from_checkpoint

    args.w = [float(x) for x in args.w.split(",")]
    args.limit_batches = float("inf") if args.limit_batches < 0 else args.limit_batches  
    
    if args.ae_noamopt !=  "" :
        args.ae_noamopt = "d_model=%s,%s"%(args.transformer_model_size, args.ae_noamopt)
                
    # run the experiments
    main(args)