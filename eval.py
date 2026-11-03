from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import cv2

class Evaluator:    
    def __init__(self):
        self.cumulative_psnr = 0.0
        self.cumulative_ssim = 0.0
        self.curr_num_evaluations = 0
        
    def get_psnr(self):
        return self.cumulative_psnr
        
    def get_ssim(self):
        print(f"returning ssim with num_evaluations {self.curr_num_evaluations}")
        return self.cumulative_ssim

    def update_psnr_and_ssim_batch(self, imgs, theta):
        '''
        Note: this method is for 2 videos only
        inputs:
        t frames from all input videos: pytorch   [num_videos=2, 3, H, W]
        thetas: forward thetas for panorama: pytorch [num_videos, 3, 3]
        
        '''
        
        # for every two imgs in the batch, calculate the psnr and ssim 
        # and update the cumulative psnr and ssim
        
        img1 = imgs[0]
        img2 = imgs[1]
        theta_21 = theta.clone().detach()
                
        psnr, ssim = self.calc_psnr_and_ssim(
                    img1, img2, theta_21
                )
                
        if psnr != 0 and ssim != 0:
            self.cumulative_psnr = (psnr + self.cumulative_psnr*self.curr_num_evaluations) / (self.curr_num_evaluations + 1)
            self.cumulative_ssim = (ssim + self.cumulative_ssim*self.curr_num_evaluations) / (self.curr_num_evaluations + 1)
            self.curr_num_evaluations += 1


    def calc_psnr_and_ssim(self, img1, img2, theta21):
        """
        Inputs:
            img0: torch tensor shape [3, H, W], reference image, range ~[0..1]
            img1: torch tensor shape [3, H, W], second image, range ~[0..1]
            theta21: torch tensor [3, 3], homography mapping image1 to align with image0
        
        Returns: psnr, ssim
        """
        img_h, img_w = img1.shape[1], img2.shape[2]
        img1_np = img1.permute(1, 2, 0).cpu().numpy()  # (H,W,C)
        img1_np = (img1_np * 255.0).clip(0, 255).astype(np.uint8)

        img2_np = img2.permute(1, 2, 0).cpu().numpy()  # (H,W,C)
        img2_np = (img2_np * 255.0).clip(0, 255).astype(np.uint8)

        theta21_np = theta21.detach().cpu().numpy().astype(np.float32)

        warped_img2 = cv2.warpPerspective(img2_np, theta21_np, (img_w, img_h))
        mask = np.any(warped_img2 > 0, axis=2).astype(np.float32)
        mask_3ch = mask[..., None]  # shape [H,W,1]

        masked_ref = img1_np * mask_3ch
        masked_warp = warped_img2 * mask_3ch

        if mask.sum() < 1:
            return 0.0, 0.0

        # Compute PSNR + SSIM
        psnr_val = peak_signal_noise_ratio(
            masked_ref, masked_warp, data_range=255
        )
        ssim_val = structural_similarity(
            masked_ref, masked_warp, data_range=255, channel_axis=2
        )

        return psnr_val, ssim_val
