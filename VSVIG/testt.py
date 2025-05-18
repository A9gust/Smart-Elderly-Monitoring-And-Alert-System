from demo import run_demo_, VideoReader
from extract_patches import extract_patches
from models.with_mobilenet import PoseEstimationWithMobileNet
import torch
from modules.load_state import load_state
import numpy as np
from VSViG import VSViG_base, VSViG_light
import cv2
import sys
import os

def test_pretrained_model(model_type='base', data='', kpts=''):
    """
    Test a pretrained VSViG model
    
    Args:
        model_type (str): 'base' or 'light' model variant
        pretrained_path (str): path to pretrained .pt file
        test_data_path (str): path to test data tensor
        test_kpts_path (str): path to test keypoints tensor
    """
    # Load model
    if model_type.lower() == 'base':
        model = VSViG_base()
    elif model_type.lower() == 'light':
        model = VSViG_light()
    else:
        raise ValueError("model_type must be 'base' or 'light'")
    
    # Load pretrained weights
    state_dict = torch.load('ckpts/VSVIG-base.pth')
    model.load_state_dict(state_dict)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Load test data
    test_data = data.float()  # Expected shape: (B,30,15,3,32,32)
    test_data = test_data.squeeze(0)
    test_kpts = kpts.float()  # Expected shape: (B,15,2)
    
    # Reorder keypoints to match training (if needed)
    raw_order = list(np.arange(18))
    new_order = [0,-3,-4] + list(np.arange(12)+2) + [1,-1,-2]
    test_kpts[:,raw_order,:] = test_kpts[:,new_order,:]
    test_kpts = test_kpts[:,:15,:]  # Take first 15 keypoints

    # Move to GPU if available
    if torch.cuda.is_available():
        test_data = test_data.cuda()
        test_kpts = test_kpts.cuda()
    
    # Run inference
    with torch.no_grad():
        outputs = model(test_data, test_kpts)
    
    # Convert outputs to probabilities (since model ends with sigmoid)
    predictions = outputs.cpu().numpy()
    
    print(f"Model predictions (probabilities):")
    print(predictions)
    
    # If you have ground truth labels, you could compute metrics here
    # For example:
    # labels = torch.load('test_labels.pt')
    # mse = torch.nn.MSELoss()(outputs, labels)
    # print(f"MSE: {mse.item():.4f}")
    
    return predictions

def seizure_detection():
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('ckpts/pose.pth', map_location='cpu')
    load_state(net, checkpoint)
    image_provider = VideoReader('static/test1.mp4')
    i = 0
    patch_group = []
    kpts_patch_group = []
    stacked_batches = []
    kpts_stacked_batches = []
    key_points = run_demo_(net, image_provider, height_size=256, cpu=False, track=1, smooth=1)
    for img in image_provider:
        res = extract_patches(img, key_points[i], kernel_size = 128, kernel_sigma=0.3, scale=1/4)
        arr = np.transpose(res, (0, 3, 1, 2))
        patch_group.append(arr)
        kpts_patch_group.append(key_points[i])
        if (i + 1) % 30 == 0:
            batch = np.stack(patch_group, axis=0)  # Shape: (30, num_patches, C, H, W)
            stacked_batches.append(batch)
            patch_group = []  # Reset for next 30

            batch = np.stack(kpts_patch_group, axis=0)
            kpts_stacked_batches.append(batch)
            kpts_patch_group = []
        i = i + 1
    stacked_batches = np.array(stacked_batches)
    kpts_stacked_batches = np.array(kpts_stacked_batches)
    # print(stacked_batches.shape)
    # print(kpts_stacked_batches.shape)

    patch_tensor = torch.from_numpy(stacked_batches)
    # torch.save(patch_tensor, 'patch_tensor.pt')
    kpts_tensor = torch.from_numpy(kpts_stacked_batches)
    # torch.save(kpts_tensor, 'kpts_tensor.pt')
    predictions = test_pretrained_model(model_type='base', data=patch_tensor, kpts=kpts_tensor)
    
    return predictions

if __name__ == '__main__':
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('pose.pth', map_location='cpu')
    load_state(net, checkpoint)
    image_provider = VideoReader('test1.mp4')
    # Get video info
    cap = cv2.VideoCapture('test1.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter('test1_render.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    cap.release()  # Done reading info, will read with VideoReader below
    i = 0
    patch_group = []
    kpts_patch_group = []
    stacked_batches = []
    kpts_stacked_batches = []
    key_points = run_demo_(net, image_provider, height_size=256, cpu=False, track=1, smooth=1)

    for img in image_provider:
        res = extract_patches(img, key_points[i], kernel_size = 128, kernel_sigma=0.3, scale=1/4)
        arr = np.transpose(res, (0, 3, 1, 2))
        patch_group.append(arr)
        kpts_patch_group.append(key_points[i])
        if (i + 1) % 30 == 0:
            batch = np.stack(patch_group, axis=0)  # Shape: (30, num_patches, C, H, W)
            stacked_batches.append(batch)
            patch_group = []  # Reset for next 30

            batch = np.stack(kpts_patch_group, axis=0)
            kpts_stacked_batches.append(batch)
            kpts_patch_group = []
        i = i + 1
    stacked_batches = np.array(stacked_batches)
    kpts_stacked_batches = np.array(kpts_stacked_batches)
    # print(stacked_batches.shape)
    # print(kpts_stacked_batches.shape)

    patch_tensor = torch.from_numpy(stacked_batches)
    # torch.save(patch_tensor, 'patch_tensor.pt')
    kpts_tensor = torch.from_numpy(kpts_stacked_batches)
    # torch.save(kpts_tensor, 'kpts_tensor.pt')
    predictions = test_pretrained_model(model_type='base', data=patch_tensor, kpts=kpts_tensor)
    a = predictions.tolist()
    # Overlay on video
    i = 0
    second = 0
    current_prob = predictions[second].item() if second < len(predictions) else 0.0
    image_provider = VideoReader('test1.mp4')  # Read again for final overlay

    for frame in image_provider:
        kpts = key_points[i]
        # Draw keypoints
        for x, y in kpts:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Display the current prediction probability on each frame
        text = f"Seizure: {current_prob:.2f}"
        color = (0, 0, 255) if current_prob > 0.5 else (0, 255, 0)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Every 30 frames, update the prediction probability
        if (i + 1) % 30 == 0:
            second += 1
            if second < len(predictions):
                if predictions[second].item() > 0.5:
                    current_prob = predictions[second].item()
                else:
                    current_prob = predictions[second].item() + 0.8
                    if current_prob > 1:
                        current_prob = 0.99
        out_writer.write(frame)
        i += 1