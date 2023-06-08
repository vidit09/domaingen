import numpy as np
import os
import tempfile
from collections import OrderedDict, defaultdict

from detectron2.utils import comm
from detectron2.evaluation import  PascalVOCDetectionEvaluator
from detectron2.evaluation.pascal_voc_evaluation import voc_eval

class CustomPascalVOCDetectionEvaluator(PascalVOCDetectionEvaluator):
    def __init__(self,dataset_name):
        super().__init__(dataset_name)

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )


        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}

        clsaps = ','.join(['{:.2f}'.format(a) for a in aps[50]])
        self._logger.info("classwise ap {}".format(clsaps))

        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
        return ret
