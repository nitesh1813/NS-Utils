local common = dofile('../common2.lua')
local dataloader = dofile('dataloader.lua')
local model = dofile('model.lua')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('oc')
require 'cutorch'

function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end
local opt = {}
-- opt.ex_data_root = 'preprocessed'
opt.ex_data_ext = 'oc'
opt.out_root = 'Results'
opt.n_classes = 14
opt.batch_size = 1

opt.weightDecay = 0.0001
opt.learningRate = 1e-3
opt.n_epochs = 100
opt.learningRate_steps = {}
opt.learningRate_steps[15] = 0.1
opt.optimizer = optim['adam']

channels={1,16,32,64,128,256,128,64,32,16}
local n_grids = 0   

net = model.get_unet(opt)
-- net:add(OctreeLogSoftMax)
-- net:cuda()

opt.net=net
-- common.net_he_init(opt.net)
opt.net:cuda()

imgs_list = lines_from("/media/McVeighLab/projects/SNitesh/datasetsall-classes-all-phases-1.5/ALL_SEGS/img_lists_1.5") 

segs_list = lines_from("/media/McVeighLab/projects/SNitesh/datasetsall-classes-all-phases-1.5/ALL_SEGS/seg_lists_1.5")

print(#imgs_list)
-- img_list_test = { unpack( imgs_list, 198, 209 ) }
-- seg_list_test = { unpack( segs_list, 198, 209 ) }
-- print(img_list_test[1])
print('[INFO] train test split')
local t = torch.Timer()
opt.criterion = oc.OctreeCrossEntropyCriterion()
-- opt.criterion = cudnn.VolumetricCrossEntropyCriterion()
opt.criterion:cuda()

print('[INFO] get train_labels')
local t = torch.Timer()

print('[INFO] get train_labels took '..t:time().real..'[s]')

-- print(#img_list_test)
local test_data_loader = dataloader.DataLoader(imgs_list ,segs_list ,opt.batch_size  ,opt.ex_data_ext ,true)
common.test(opt,test_data_loader)
