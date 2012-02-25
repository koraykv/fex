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
    local imtens = torch.ByteTensor(imstor):resize(w,h,nchannels)
    local dimtens = torch.Tensor(nchannels,w,h):copy(imtens)
    for i=1,nchannels do
        dimtens[i]:copy(imtens:select(3,i))
    end
    dimtens:div(255)
    return dimtens:squeeze()
end

function fex.readrgb(fname)
    return readim(fname,3)
end
function fex.readrgba(fname)
    return readim(fname,4)
end
function fex.readgray(fname)
    return readim(fname,1)
end
function fex.imread(fname)
    return readim(fname)
end
local flena = paths.thisfile('lena.png')
function fex.lena()
    return fex.imread(flena)
end
