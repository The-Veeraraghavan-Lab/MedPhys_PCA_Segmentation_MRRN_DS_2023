from .base_options import BaseOptions

class SegmentationOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--out_wt', type=float, default=0.0, help='Weight for output vs. deep layer outputs')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        #self.parser.add_argument('--optimizer', type=str, default='Adam', help='which epoch to load? set to latest to use latest cached model')
        #self.parser.add_argument('--loss', type=str, default='ce', help='Loss to use? Default is combined dice and cross-entropy loss: choices are: dice, tversky, focal, soft_dsc')
        self.parser.add_argument('--model_type', type=str, default='deep', help='model type to use? Default is standard, other options are ''deep'' for deep supervision from an extra layer; ''multi'' for deep supervised classification; ''classifier'' for classification only')
        self.parser.add_argument('--model_name', type=str, default='MRRNDS_model', help='model name to test.')
        self.parser.add_argument('--nslices', type=int, default=5, help='slices for modality type')
        self.isTrain = False
        #self.parser.add_argument('--model', type=str, default='circle_gan_unet', help='multiply by a gamma every lr_decay_iters iterations')