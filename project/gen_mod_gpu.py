def infer(data_path, output_path, parameters_file, weights, save_back = False):
    # get test options
    opt = OPTIONS()
    opt.dataroot = data_path 
    opt.output_dir = output_path
    for key in parameters_file.keys():
        setattr(opt, key, parameters_file[key])
    
    str_ids = opt.gpu_ids.split(',') if isinstance(opt.gpu_ids, str) else [str(opt.gpu_ids)]
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    
    # FORCE GPU USAGE
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
        print(f"✓ Set CUDA device to: {torch.cuda.get_device_name(opt.gpu_ids[0])}")
    else:
        print("❌ No GPU configured!")
        return
    
    # OPTIMIZE SETTINGS
    opt.num_threads = 4   # Increase from 0
    opt.batch_size = 4    # Increase from 1 
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
                
                # CHECK MODEL IS ON GPU
                model_device = next(model.netG.parameters()).device
                print(f"✓ Model loaded on: {model_device}")
                if model_device.type != 'cuda':
                    print("❌ ERROR: Model is on CPU, not GPU!")
                    # Force model to GPU
                    model.netG = model.netG.cuda()
                    print("✓ Moved model to GPU")
                
            if i >= opt.num_test:
                break
            
            # FORCE DATA TO GPU
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].cuda(non_blocking=True)
            
            model.set_input(data)
            
            # VERIFY INPUT IS ON GPU (first iteration only)
            if i == 0:
                input_device = model.real_A.device
                print(f"✓ Input data on: {input_device}")
                if input_device.type != 'cuda':
                    print("❌ ERROR: Input data is on CPU!")
                    return
                
                print("✓ Starting GPU inference...")
                
                # Test a single forward pass and time it
                import time
                start_time = time.time()
                with torch.no_grad():
                    test_output = model.netG(model.real_A)
                gpu_time = time.time() - start_time
                print(f"✓ First GPU inference took: {gpu_time:.3f}s")
            
            # ACTUAL INFERENCE
            with torch.no_grad():  # Ensure no gradients
                model.test()
            
            # Process batch results
            batch_size = data["A"].size(0)
            for batch_idx in range(batch_size):
                output = model.fake_B[batch_idx:batch_idx+1].cpu()
                output_cube = convert_image_to_256_3d(output).squeeze()
                affine_matrix = data['affine'][batch_idx:batch_idx+1]
                
                if save_back:
                    folder_path = None 
                    if 'folder_path' in data.keys():
                        folder_path = data['folder_path'][batch_idx]
                    else:
                        a_path = data['A_paths'][batch_idx]
                        temp = a_path.split(os.path.sep)
                        folder_path = os.path.join(*temp[:-1])
                    
                    save_to_output(output_cube,
                               affine_matrix, 
                               data['A_paths'][batch_idx], 
                               folder_path, data['test_target_modality'][batch_idx],
                               for_segmentation = opt.for_segmentation)
                else:
                    save_to_output(output_cube,
                               affine_matrix, 
                               data['A_paths'][batch_idx], 
                               opt.output_dir, data['test_target_modality'][batch_idx],
                               for_segmentation = opt.for_segmentation)
            
            # Monitor GPU usage every 50 iterations
            if i % 50 == 0 and i > 0:
                print(f"Processed {i}/{len(dataset)} batches...")

if __name__ == "__main__":
    import yaml
    data_path = "pseudo_val_set"
    output_path = "pseudo_val_output"
    weights = "mlcube/workspace/additional_files/weights/"
    parameters_file = "mlcube/workspace/parameters.yaml"
    with open(parameters_file) as f:
        parameters = yaml.safe_load(f)
    
    print("=== Starting GPU-Optimized Inference ===")
    infer(data_path, output_path, parameters, weights, save_back=True)