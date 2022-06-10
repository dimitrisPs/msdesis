from .segmentation import dice_wbce, jaccard_wbce
from .disparity import unsupervised_disparity_loss_multiscale, l1_multiscale_loss


def get_segmentation_loss(loss:str):
    loss_request=loss.lower()
    if loss_request=='dice_wbce':
        return dice_wbce
    elif loss_request=='jaccard_wbce':
        return jaccard_wbce
    else:
        raise NotImplementedError

def get_disparity_loss(loss:str):
    loss_request=loss.lower()
    if loss_request=='multiscale_smooth_l1':
        return l1_multiscale_loss
    elif loss_request=='unsupervised_multiscale':
        return unsupervised_disparity_loss_multiscale
    else:
        raise NotImplementedError
