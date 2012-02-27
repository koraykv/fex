
local fex = {}

fex.verbose = false
torch.include('fex','dist.lua')
torch.include('fex','sift.lua')
--torch.include('fex','hog.lua')
torch.include('fex','utils.lua')
torch.include('fex','imageio.lua')
torch.include('fex','display.lua')
