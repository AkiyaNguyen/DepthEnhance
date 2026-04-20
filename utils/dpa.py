import torch
import random

def dpa(depth_map, images_s, pred_u, beta, t, T):
    ## convert depth_map to 1 channel (support both 1-channel and 3-channel depth)
    if depth_map.size(1) == 1:
        grey_depth_map = depth_map
    else:
        weights = torch.tensor([0.299, 0.587, 0.114], device=depth_map.device, dtype=depth_map.dtype).view(1, 3, 1, 1)
        grey_depth_map = (depth_map * weights).sum(dim=1, keepdim=True)  # (N, 1, H, W)
    
    def calculate_variance_hardness(grey_depth_map):

        N, _, H, W = grey_depth_map.size()
        h, w = random.choice([40, 10, 20]), random.choice([40, 10, 20])


        ## not sure about the logic
        patches = grey_depth_map.unfold(2, h, h).unfold(3, w, w)
        patch_variance = patches.var(dim=(-2, -1))
        hardness_scores = patch_variance.flatten(1)

        return hardness_scores, h, w



    hardness_scores, h, w = calculate_variance_hardness(grey_depth_map)

    N, num_patches = hardness_scores.size()

    if t<T:
       k = int(beta * (t / T) * num_patches)
    else:
        k = int(beta * num_patches)


    _, indices = torch.topk(hardness_scores, k, dim=1, largest=True)

    augmented_imgs = []
    augmented_labels = []

    for i in range(N):
        mask = torch.zeros(num_patches, dtype=torch.float32)
        mask[indices[i]] = 1.0 ## the most hardness are marked as 1
        available_indices = [j for j in range(num_patches) if mask[j] == 0] ## the rest are available for augmentation
        augmented_img = images_s[i].clone()
        augmented_label = pred_u[i].clone()


        if available_indices:
            chosen_patch = random.choice(available_indices)


            patch_h = chosen_patch // (images_s.size(3) // w)
            patch_w = chosen_patch % (images_s.size(3) // w)
            x_start = patch_w * w
            x_end = x_start + w
            y_start = patch_h * h
            y_end = y_start + h


            next_img_idx = (i + 1) % N
            augmented_img[:, y_start:y_end, x_start:x_end] = images_s[next_img_idx][:, y_start:y_end, x_start:x_end]
            augmented_label[:, y_start:y_end, x_start:x_end] = pred_u[next_img_idx][:, y_start:y_end, x_start:x_end]

        augmented_imgs.append(augmented_img)
        augmented_labels.append(augmented_label)


    augmented_imgs = torch.stack(augmented_imgs)
    augmented_labels = torch.stack(augmented_labels)

    return augmented_imgs, augmented_labels


def apa_cutmix(images_s, pred_u, beta, t, T):
    eps = 1e-10

    def calculate_pred_hardness(pred_u):
        N, C, H, W = pred_u.size()
        h, w = random.choice([40, 10, 20]), random.choice([40, 10, 20])

        patches = pred_u.unfold(2, h, h).unfold(3, w, w)
        n_h, n_w = patches.size(2), patches.size(3)

        if C == 1:
            # Binary entropy for sigmoid outputs.
            p = patches.clamp(min=eps, max=1.0 - eps)
            pixel_entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
            patch_entropy = pixel_entropy.mean(dim=(-1, -2, 1))
        else:
            # Categorical entropy for multi-class probabilities.
            p = patches.clamp(min=eps, max=1.0)
            patch_entropy = -(p * torch.log(p)).sum(dim=1).mean(dim=(-1, -2))

        hardness_scores = patch_entropy.reshape(N, -1)
        return hardness_scores, h, w, n_h, n_w

    hardness_scores, h, w, _, n_w = calculate_pred_hardness(pred_u)

    N, num_patches = hardness_scores.size()

    if t<T:
       k = int(beta * (t / T) * num_patches)
    else:
        k = int(beta * num_patches)

    _, indices = torch.topk(hardness_scores, k, dim=1, largest=True)

    augmented_imgs = []
    augmented_labels = []

    for i in range(N):
        mask = torch.zeros(num_patches, dtype=torch.float32, device=pred_u.device)
        mask[indices[i]] = 1.0 ## the most hardness are marked as 1
        available_indices = torch.where(mask == 0)[0].tolist() ## the rest are available for augmentation
        augmented_img = images_s[i].clone()
        augmented_label = pred_u[i].clone()

        if available_indices:
            chosen_patch = random.choice(available_indices)
            next_img_idx = (i + 1) % N
            patch_h = chosen_patch // n_w
            patch_w = chosen_patch % n_w
            x_start = patch_w * w
            x_end = x_start + w
            y_start = patch_h * h
            y_end = y_start + h

            augmented_img[:, y_start:y_end, x_start:x_end] = images_s[next_img_idx][:, y_start:y_end, x_start:x_end]
            augmented_label[:, y_start:y_end, x_start:x_end] = pred_u[next_img_idx][:, y_start:y_end, x_start:x_end]

        augmented_imgs.append(augmented_img)
        augmented_labels.append(augmented_label)


    augmented_imgs = torch.stack(augmented_imgs)
    augmented_labels = torch.stack(augmented_labels)

    return augmented_imgs, augmented_labels