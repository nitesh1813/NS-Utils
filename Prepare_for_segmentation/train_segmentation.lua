#!/usr/bin/env th

local common = dofile('segmentation_common.lua')
local model = dofile('model.lua')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('oc')
require 'cutorch'
-- torch.backends.cudnn.enabled = false 
local opt = {}
-- opt.ex_data_root = 'preprocessed'
opt.ex_data_ext = 'oc'
opt.out_root = 'Results'

opt.n_classes = 14
opt.batch_size = 2		
opt.weightDecay = 0.001
opt.learningRate = 1e-3
opt.n_epochs = 70
opt.learningRate_steps = {}
opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']

function create_DataParallelNetwork(model)
	print('converting module to nn.DataParallelTable')
	local model_single = model
	model = nn.DataParallelTable(1)
	for k,v in ipairs(GPU_LIST) do
	    print ('Device ',v)
	cutorch.setDevice(v)
	model:add(model_single:clone():cuda(), v)
	end
	cutorch.setDevice(1) -- *** decide on the default GPU ***
	return model
end
function add_convolution_layer( input_size,feature_size,input )
   conv1 = oc.OctreeConvolutionMM(input_size,feature_size, 0) (input)
   relu1 = oc.OctreeReLU(true) (conv1)
   conv2 = oc.OctreeConvolutionMM(feature_size, feature_size, 0) (relu1)
   relu2 = oc.OctreeReLU(true)(conv2)
   return relu2
   

end
net = model.get_seg_net(opt)
-- net:cuda()
--weights= torch.FloatTensor({0.9,1,1,1,1,1,1,1,1,1,1,1})
-- weights = torch.FloatTensor(1.83464309e-03,1.57653658e-01,1.68126137e-01,2.09311726e+00,6.24825240e-01,4.02656307e-01,6.51944090e+00,4.18466961e+00,6.86760152e+00,5.73249122e+00,1.00000000e+00})
GPU_LIST = torch.range(1,2):totable()

-- net = create_DataParallelNetwork(net)
-- net:cuda()
-- print(gpu_table)
-- local dpt = nn.DataParallelTable(1,true):add(net,gpu_table):threads(function () require 'cudnn' cudnn.benchmark = true end)
-- net = dpt:cuda()
opt.net=net
common.net_he_init(opt.net)
opt.net:cuda()
opt.criterion = oc.OctreeCrossEntropyCriterion()
-- opt.criterion = cudnn.VolumetricCrossEntropyCriterion()
opt.criterion:cuda()
-- print(cutorch.getMemoryUsage(1))
-- print(torch.cuda.memory_allocated())
common.segmentation_worker(opt)

