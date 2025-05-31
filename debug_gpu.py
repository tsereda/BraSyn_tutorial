def infer_with_gpu_fix(data_path, output_path, parameters_file, weights, save_back=False):
    opt = OPTIONS()
    opt.dataroot = data_path 
    opt.output_dir = output_path
    for key in parameters_file.keys():
        setattr(opt, key, parameters_file[key])
    
    # FORCE GPU USAGE - this is the key fix
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    
    # Force CUDA device
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])
        print(f"Using GPU: {torch.cuda.get_device_name(opt.gpu_ids[0])}")
    else:
        print("WARNING: No GPU available or not configured!")
        return
    
    # Optimize settings
    opt.num_threads = 4
    opt.batch_size = 4  # Start with 4, increase if memory allows
    opt.serial_batches = True
    opt.paired = True
    opt.no_flip = True
    opt.display_id = -1
    opt.checkpoints_dir = weights
    os.makedirs(opt.output_dir, exist_ok=True)
    
    if not opt.single_folder:
        opt.phase = 'test'
        dataset = create_dataset(opt)
        model = create_model(opt)
        print("dataset size: ", len(dataset), 'dataset root: ', dataset.dataset.root)
        
        for i, data in enumerate(tqdm(dataset, total=min(opt.num_test, len(dataset)), desc='Testing')):
            if i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
                if opt.eval:
                    model.eval()
                
                # EXPLICIT GPU CHECK - verify model is on GPU
                device = next(model.netG.parameters()).device
                print(f"Model device: {device}")
                if device.type != 'cuda':
                    print("ERROR: Model is not on GPU!")
                    return
                    
            if i >= opt.num_test:
                break
            
            # EXPLICIT DATA TO GPU - ensure input data is on GPU
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].cuda(non_blocking=True)
            
            model.set_input(data)
            
            # Verify input is on GPU
            if i == 0:
                input_device = model.real_A.device
                print(f"Input device: {input_device}")
                if input_device.type != 'cuda':
                    print("ERROR: Input data is not on GPU!")
                    return
            
            model.test()
            
            # Process outputs (rest of your code)
            output = model.fake_B.cpu()
            output_cube = convert_image_to_256_3d(output).squeeze()
            affine_matrix = data['affine']
            
            if save_back:
                folder_path = None 
                if 'folder_path' in data.keys():
                    folder_path = data['folder_path'][0]
                else:
                    a_path = data['A_paths'][0]
                    temp = a_path.split(os.path.sep)
                    folder_path = os.path.join(*temp[:-1])
                
                save_to_output(output_cube,
                           affine_matrix, 
                           data['A_paths'][0], 
                           folder_path, data['test_target_modality'][0],
                           for_segmentation=opt.for_segmentation)
            else:
                save_to_output(output_cube,
                           affine_matrix, 
                           data['A_paths'][0], 
                           opt.output_dir, data['test_target_modality'][0],
                           for_segmentation=opt.for_segmentation)