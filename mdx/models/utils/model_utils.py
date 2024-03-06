from mdx import models


def get_model_class(model_type):
    r"""Get model.

    Args:
        model_type: str, e.g., 'ResUNet143_DecouplePlusInplaceABN'

    Returns:
        nn.Module
    """
    if model_type == "ResUNet143_DecouplePlusInplaceABN_ISMIR2021":
        from mdx.models.resunet_ismir2021 import (
            ResUNet143_DecouplePlusInplaceABN_ISMIR2021,
        )

        return ResUNet143_DecouplePlusInplaceABN_ISMIR2021

    elif model_type == "UNet":
        from mdx.models.unet import UNet

        return UNet

    elif model_type == "UNetSubbandTime":
        from mdx.models.unet_subbandtime import UNetSubbandTime

        return UNetSubbandTime

    elif model_type == "ResUNet143_Subbandtime":
        from mdx.models.resnet_subbandtime import ResUNet143_Subbandtime

        return ResUNet143_Subbandtime

    elif model_type == "MobileNet_Subbandtime":
        from mdx.models.mobilenet_subbandtime import MobileNet_Subbandtime

        return MobileNet_Subbandtime

    elif model_type == "MobileTiny_Subbandtime":
        from mdx.models.mobiletiny_subbandtime import MobileTiny_Subbandtime

        return MobileTiny_Subbandtime

    elif model_type == "ResUNet143_DecouplePlus":
        from mdx.models.resunet import ResUNet143_DecouplePlus

        return ResUNet143_DecouplePlus

    elif model_type == "ConditionalUNet":
        from mdx.models.conditional_unet import ConditionalUNet

        return ConditionalUNet

    elif model_type == "LevelRNN":
        from mdx.models.levelrnn import LevelRNN

        return LevelRNN

    elif model_type == "WavUNet":
        from mdx.models.wavunet import WavUNet

        return WavUNet

    elif model_type == "WavUNetLevelRNN":
        from mdx.models.wavunet_levelrnn import WavUNetLevelRNN

        return WavUNetLevelRNN

    elif model_type == "TTnet":
        from mdx.models.ttnet import TTnet

        return TTnet

    elif model_type == "TTnetNoTransformer":
        from mdx.models.ttnet_no_transformer import TTnetNoTransformer

        return TTnetNoTransformer

    elif model_type == "JiafengCNN":
        from mdx.models.ttnet_jiafeng import JiafengCNN

        return JiafengCNN

    elif model_type == "JiafengTTNet":
        from mdx.models.ttnet_jiafeng import JiafengTTNet

        return JiafengTTNet

    elif model_type == "ResUNet143FC_Subbandtime":
        from mdx.models.resunet_subbandtime2 import ResUNet143FC_Subbandtime

        return ResUNet143FC_Subbandtime

    elif model_type == "AmbisonicToBinaural_UNetSubbandtimePhase":
        from mdx.models.ambisonic_to_binaural import (
            AmbisonicToBinaural_UNetSubbandtimePhase,
        )

        return AmbisonicToBinaural_UNetSubbandtimePhase

    elif model_type == "AmbisonicToBinaural_ResUNetSubbandtimePhase":
        from mdx.models.ambisonic_to_binaural import (
            AmbisonicToBinaural_ResUNetSubbandtimePhase,
        )

        return AmbisonicToBinaural_ResUNetSubbandtimePhase

    elif model_type == "MobileNetSubbandTime":
        from mdx.models.mobilenet_subbandtime import MobileNetSubbandTime

        return MobileNetSubbandTime

    elif model_type == "WrapperDemucs":
        from mdx.models.demucs.demucs import WrapperDemucs

        return WrapperDemucs

    elif model_type == "WrapperHDemucs":
        from mdx.models.demucs.hdemucs import WrapperHDemucs

        return WrapperHDemucs

    else:
        raise NotImplementedError("{} not implemented!".format(model_type))
