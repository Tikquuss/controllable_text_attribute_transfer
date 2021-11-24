import argparse
import time

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")
    
def add_log(args, ss):
    now_time = time.strftime("[%Y-%m-%d %H:%M:%S]: ", time.localtime())
    print(now_time + ss)
    with open(args.log_file, 'a') as f:
        f.write(now_time + str(ss) + '\n')

def add_output(args, ss):
    with open(args.output_file, 'a') as f:
        f.write(str(ss) + '\n')

def write_text_z_in_file(args, text_z_prime) :
    with open(args.output_file, 'w') as f:
        keys = list(text_z_prime.keys())
        K_ =  len(keys)
        L = len(text_z_prime[keys[0]])
        for i in range(L) :
            item = [text_z_prime[k][i] for k in keys]
            for j in range(len(item[0])) :
                f.writelines(["%s : %s\n"% (keys[k], item[k][j]) for k in range(K_)])
                f.write("\n")