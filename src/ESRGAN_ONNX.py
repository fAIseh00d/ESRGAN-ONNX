import numpy as np
from PIL import Image

class ESRGAN:

    def __init__(self, model_file=None, session=None, tile_size=128, prepad=8, scale=None):
        self.model_file = model_file
        self.session = session
        self.tile_size = tile_size
        self.prepad = prepad
        self.scale = scale
        self.model_input = self.session.get_inputs()[0].name
        if not self.scale:
            self.scale = self._get_scale()
        print('Upscaling factor = ', self.scale)


    def _tile_preprocess(self, img):
        input_data = np.array(img).transpose(2, 0, 1)
        img_data = input_data.astype('float32') / 255.0
        padded_size = self.tile_size + self.prepad*2
        norm_img_data = img_data.reshape(1, 3, padded_size, padded_size).astype('float32')
        return norm_img_data


    def _into_tiles(self, image_array):
        img = Image.fromarray(image_array).convert('RGB')
        self.width, self.height = img.size
        tile_size = self.tile_size
        prepad = self.prepad
        self.num_width = int(np.ceil(self.width/tile_size))
        self.num_height = int(np.ceil(self.height/tile_size))
        self.pad_width = self.num_width*tile_size
        self.pad_height = self.num_height*tile_size
        pad_img = Image.new("RGB", (self.pad_width, self.pad_height))
        pad_img.paste(img)
        tiles = []
        for i in range(self.num_height):
            for j in range(self.num_width):
                box = [j*tile_size, i*tile_size, (j+1)*tile_size, (i+1)*tile_size]
                box = [box[0]-prepad, box[1]-prepad, box[2]+prepad, box[3]+prepad]
                tiles.append(self._tile_preprocess(pad_img.crop(tuple(box))))
        return tiles


    def _into_whole(self, tiles):
        scaled_tile = self.scale * self.tile_size
        scaled_pad = self.scale * self.prepad
        out_img = Image.new("RGB", (self.pad_width*self.scale, self.pad_height*self.scale))
        paste_cnt = 0
        for i in range(self.num_height):
            for j in range(self.num_width):
                box = (scaled_pad,scaled_pad,scaled_pad+scaled_tile,scaled_pad+scaled_tile)
                tile = tiles[paste_cnt].resize((scaled_pad*2+scaled_tile,scaled_pad*2+scaled_tile))
                paste_pos = (j*scaled_tile, i*scaled_tile)
                out_img.paste(tile.crop(box), paste_pos)
                paste_cnt += 1
        out_img = out_img.crop((0, 0, self.width*self.scale, self.height*self.scale))
        return np.array(out_img)


    def get(self, image_array):
        input_tiles = self._into_tiles(image_array)
        output_tiles = []
        for tile in input_tiles:
            result = self.session.run([], {self.model_input: tile})[0][0]
            result = np.clip(result.transpose(1, 2, 0), 0, 1) * 255.0
            output_tiles.append(Image.fromarray(result.round().astype(np.uint8)))
        return self._into_whole(output_tiles)

    def _get_scale(self):
        # Create a test BGR image with 1px by 1px size filled with float32 white 
        input_temp = np.full((1, 1, 3), (1.0, 1.0, 1.0), dtype=np.float32)
        # Transpose the dimensions of the input tensor (numpy image array to esrgan compatible image)
        input_temp = np.transpose(input_temp, (2, 0, 1))
        # Add a new dimension to the input tensor (batch_size)
        input_temp = np.expand_dims(input_temp, axis=0)
        output_temp = self.session.run([], {self.model_input: input_temp})[0][0]
        scale_factor = int(output_temp.shape[-1]/input_temp.shape[-1])
        return scale_factor