#coding = "utf-8"
import os
import util

parser = util.parser
args = parser.parse_args()
os.chdir(args.save_path)
for i in range(args.training_times):
    print('------------------------------')
    print("GSAPool Training Control Shell")
    print('------------------------------')
    print('Training Dataset:   ', args.dataset)
    print('Pooling Layer Type: ',args.pooling_layer_type)
    print('Feature Fusion Type:',args.feature_fusion_type)
    print('------------------------------')
    os.system("python main.py")
with open(os.path.join(args.save_path, 'result.txt'), 'a') as f:
    f.write('\r\n')
    



