from deploy.tracker import kcftracker
import numpy as np

class KCF:
    def __init__(self):
        self.trackers = []

    def update(self, frame):
        result = []
        for i, item in enumerate(self.trackers):
            tracker = item["tracker"]
            boundingbox = tracker.update(frame)
            temp_dict = {
                "id": i,
                "boundingbox": boundingbox,
                "label": item["label"]
            }
            result.append(temp_dict)
        return result


        # to_del = []
        # for i, trk in enumerate(trks):
        #     pos = self.trackers[i].detect()
        #     trk[:] = [pos[0], pos[1], pos[2], pos[3]. 0]
        #     if np.any(np.isnan(pos))
        #         to_del.append(t)
        #
        # for t in reversed(to_del):
        #     self.trackers.pop(t)
        #
        # if dets != []:
