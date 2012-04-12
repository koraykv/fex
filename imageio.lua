local convertcmd = io.popen('which convert','r'):read('*all')
if #convertcmd == 0 then
    error('Could not find convert command line program from imagemagick...')
end

-- this can read all formats that imagemagick can read
local function readim(fname,nchannels)
    local cid = 'identify ' .. tostring(fname)
    local id = io.popen(cid):read('*l')
    if not id then error('could not get the size of ' .. fname) end
    local w,h = id:match('%w+ %w+ (%d+)x(%d+)')
    w=tonumber(w)
    h=tonumber(h)
    local ext
    if not w or not h then error ('Invalid width, height', tostring(w), tostring(h)) end
    if not nchannels then
        local colorspace = io.popen('identify -format %[colorspace] ' .. fname):read('*l')
        if colorspace:lower() == 'gray'then
            ext = '.gray'
            nchannels = 1
        elseif colorspace:lower() == 'rgb' then
            ext = '.rgb'
            nchannels = 3
        elseif colorspace:lower() == 'rgba' then
            ext = '.rgba'
            nchannels = 4
        else
            ext = '.rgb'
            nchannels = 3
        end
    else
        ext = '.rgb'
        if nchannels == 1 then ext = '.gray' end
        if nchannels == 3 then ext = '.rgb' end
        if nchannels == 4 then ext = '.rgba' end
    end
    local tmpf = os.tmpname() .. ext
    local cmd = 'convert -depth 8 -size ' .. w .. 'x' .. h .. ' ' .. fname .. ' ' .. tmpf
    local ss = io.popen(cmd,'r'):read('*all')
    if #ss > 0 then error(s) end
    local imstor = torch.DiskFile(tmpf,'r'):binary():readByte(w*h*nchannels)
    local imtens = torch.ByteTensor(imstor):resize(h,w,nchannels):transpose(1,3):transpose(2,3)
    local dimtens = torch.Tensor(nchannels,h,w):copy(imtens)
    dimtens:div(255)
    if nchannels == 1 then dimtens:resize(h,w) end
    return dimtens
end

-- this can write all formats that imagemagick can write
local function writeim(im,fname)
    local w,h,nch = im:size(2),im:size(1),im:size(3)
    local ext
    if nch == 1 then
        ext = '.gray'
    elseif nch == 3 then
        ext = '.rgb'
    elseif nch == 4 then
        ext = '.rgba'
    else
        error('Number of channels of an image can be 1 (gray), 3 (rgb) or 4 (rgba)')
    end
    local tmpf = os.tmpname() .. ext
    local imstor = torch.DiskFile(tmpf,'w'):binary()
    imstor:writeByte(im:storage())
    imstor:close()
    local cmd = 'convert -depth 8 -size ' .. w .. 'x' .. h .. ' ' .. tmpf .. ' ' .. fname
    local ss = io.popen(cmd,'r'):read('*all')
    if #ss > 0 then error(s) end
end

local function readrgb(fname)
   return readim(fname,3)
end
local function readrgba(fname)
   return readim(fname,4)
end
local function readgray(fname)
   return readim(fname,1)
end
function fex.imread(fname,type)
   if not type then
      return readim(fname)
   end
   if type == 'gray' then
      return readgray(fname)
   elseif type == 'rgb' then
      return readrgb(fname)
   elseif type == 'rgba' then
      return readrgba(fname)
   end
   return readim(fname)
end
      
function fex.imwrite(fname,im)
    local imc = im:clone()
    if imc:max() > 1 then imc:div(imc:max()) end
    imc:mul(255)
    if imc:dim() == 2 then imc:resize(1,imc:size(1),imc:size(2)) end
    local imb = imc:byte():transpose(1,3):transpose(1,2):clone()
    writeim(imb,fname)
end

local flena = paths.thisfile('lena.png')
function fex.lena()
    return fex.imread(flena)
end
