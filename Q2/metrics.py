import torch

def calculate_metrics(preds, targets, num_classes=23):
    preds = torch.argmax(preds, dim=1)
    iou_list = []
    dice_list = []

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = pred_inds.sum() + target_inds.sum() - intersection

        if union > 0:
            iou_list.append((intersection / union).item())
            dice = (2.0 * intersection) / (pred_inds.sum() + target_inds.sum())
            dice_list.append(dice.item())

    mIoU = sum(iou_list) / len(iou_list) if iou_list else 0.0
    mDice = sum(dice_list) / len(dice_list) if dice_list else 0.0
    
    return mIoU, mDice