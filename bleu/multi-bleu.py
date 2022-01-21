from bleu import corpus_bleu
import argparse
import os
import codecs

import stat
import subprocess
import warnings

BLEU_SCRIPT_PATH=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')

def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    # our
    if os.name == 'nt' : # windows os
        command = "perl " + command

    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)

    #global stop_threads
    #stop_threads = False
    #threading.Thread(target = thread_target,  kwargs={"cmd" : "taskkill /f /im notepad.exe", "wait" : 5, "timeout" : None}).start()
    result = p.communicate()[0].decode("utf-8")
    #stop_threads = True
    
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        warnings.warn('Impossible to parse BLEU score! "%s"' % result)
        return -1

def file_exist(pf):
    if os.path.isfile(pf):
        return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate BLEU scores for the input hypothesis and reference files")
    parser.add_argument(
        "-hyp",
        nargs=1,
        dest="pf_hypothesis",
        type=str,
        help="The path of the hypothesis file.")
    parser.add_argument(
        "-ref",
        nargs='+',
        dest="pf_references",
        type=str,
        help="The path of the references files.")
    args = parser.parse_args()

    if args.pf_hypothesis!=None or args.pf_references!=None:
        if args.pf_hypothesis==None:
            raise Exception("Missing references files...")
        if args.pf_references==None:
            raise Exception("Missing hypothesis files...")
        
    n = None
    data = []
    for pf in args.pf_hypothesis+args.pf_references:
        if not file_exist(pf):
            raise Exception("File Not Found: %s"%(pf))
        f = codecs.open(pf, encoding="utf-8")
        data.append(f.readlines())
        if n==None:
            n = len(data[-1])
        elif n != len(data[-1]):
            raise Exception("Not parrallel: %s %d-%d"%(pf, n, len(data[-1])))
        f.close()
        
    hyp_data = data[0]
    ref_data = list(map(list, zip(*data[1:])))
        
    perl = False
    if not perl :
        bleu, addition = corpus_bleu(hyp_data, ref_data)
        print("BLEU = %.2f, %.1f/%.1f/%.1f/%.1f (BP=%.3f, ratio=%.3f, hyp_len=%d, ref_len=%d)"%(bleu[0]*100, bleu[1]*100, bleu[2]*100, bleu[3]*100, bleu[4]*100, addition[0], addition[1], addition[2], addition[3]))
    else :
        if len(args.pf_references) > 1 :
            c = os.path.abspath(os.path.dirname(__file__))
            hyp_path = os.path.join(c, "tmp_hyp")
            ref_path = os.path.join(c, "tmp_ref")
            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hyp_data) + '\n')
            # export sentences to hypothesis file / restore BPE segmentation
            with open(ref_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join([x[0] for x in ref_data]) + '\n')
        else :
            hyp_path = args.pf_hypothesis[0]
            ref_path = args.pf_references[0]
            
        # chmod +x ${BLEU_SCRIPT_PATH}
        # https://stackoverflow.com/a/12792002/11814682
        st = os.stat(BLEU_SCRIPT_PATH)
        os.chmod(BLEU_SCRIPT_PATH, st.st_mode | stat.S_IEXEC)
        bleu = eval_moses_bleu(ref=ref_path, hyp=hyp_path)
        print("BLEU = %.2f"%bleu)