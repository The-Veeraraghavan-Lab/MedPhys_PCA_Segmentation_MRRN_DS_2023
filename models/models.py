
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'MRRN_Segmentor':
        from .MRI_model import MRRN_Segmentor 
        model = MRRN_Segmentor() #cycle_gan_unet_ct_seg_baseline()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    
    print (opt.model)
    print (model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model