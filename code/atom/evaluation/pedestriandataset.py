import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text


class PedestrianDataset(BaseDataset):
    """ Pedestrian dataset

    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.pedestrian_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: Pedestrian has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1, 1)
            y1 = np.amin(gt_y_all, 1).reshape(-1, 1)
            x2 = np.amax(gt_x_all, 1).reshape(-1, 1)
            y2 = np.amax(gt_y_all, 1).reshape(-1, 1)

            ground_truth_rect = np.concatenate((x1, y1, x2 - x1, y2 - y1), 1)

        return Sequence(sequence_info['name'], frames, 'pedestrian', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Airport_ce", "path": "Airport_ce/img", "startFrame": 1, "endFrame": 148, "nz": 4, "ext": "jpg",
             "anno_path": "Airport_ce/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "BlurBody", "path": "BlurBody/img", "startFrame": 1, "endFrame": 334, "nz": 4, "ext": "jpg",
             "anno_path": "BlurBody/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Busstation_ce1", "path": "Busstation_ce1/img", "startFrame": 1, "endFrame": 363, "nz": 4, "ext": "jpg",
             "anno_path": "Busstation_ce1/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Busstation_ce2", "path": "Busstation_ce2/img", "startFrame": 6, "endFrame": 400, "nz": 4, "ext": "jpg",
             "anno_path": "Busstation_ce2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Crossing", "path": "Crossing/img", "startFrame": 1, "endFrame": 120, "nz": 4, "ext": "jpg",
             "anno_path": "Crossing/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Crowds", "path": "Crowds/img", "startFrame": 1, "endFrame": 347, "nz": 4, "ext": "jpg",
             "anno_path": "Crowds/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "David3", "path": "David3/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg",
             "anno_path": "David3/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Girl2", "path": "Girl2/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg",
             "anno_path": "Girl2/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Human2", "path": "Human2/img", "startFrame": 1, "endFrame": 1128, "nz": 4, "ext": "jpg",
             "anno_path": "Human2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human3", "path": "Human3/img", "startFrame": 1, "endFrame": 1698, "nz": 4, "ext": "jpg",
             "anno_path": "Human3/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human4_2", "path": "Human4/img", "startFrame": 1, "endFrame": 667, "nz": 4, "ext": "jpg",
             "anno_path": "Human4/groundtruth_rect.2.txt",
             "object_class": "person"},
            {"name": "Human5", "path": "Human5/img", "startFrame": 1, "endFrame": 713, "nz": 4, "ext": "jpg",
             "anno_path": "Human5/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human6", "path": "Human6/img", "startFrame": 1, "endFrame": 792, "nz": 4, "ext": "jpg",
             "anno_path": "Human6/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human7", "path": "Human7/img", "startFrame": 1, "endFrame": 250, "nz": 4, "ext": "jpg",
             "anno_path": "Human7/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human8", "path": "Human8/img", "startFrame": 1, "endFrame": 128, "nz": 4, "ext": "jpg",
             "anno_path": "Human8/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human9", "path": "Human9/img", "startFrame": 1, "endFrame": 305, "nz": 4, "ext": "jpg",
             "anno_path": "Human9/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Jogging_1", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg",
             "anno_path": "Jogging/groundtruth_rect.1.txt",
             "object_class": "person"},
            {"name": "Jogging_2", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg",
             "anno_path": "Jogging/groundtruth_rect.2.txt",
             "object_class": "person"},
            {"name": "Pedestrian", "path": "Pedestrian/img", "startFrame": 1, "endFrame": 140, "nz": 8, "ext": "jpg",
             "anno_path": "Pedestrian/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Railwaystation_ce", "path": "Railwaystation_ce/img", "startFrame": 1, "endFrame": 413, "nz": 4, "ext": "jpg",
             "anno_path": "Railwaystation_ce/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Subway", "path": "Subway/img", "startFrame": 1, "endFrame": 175, "nz": 4, "ext": "jpg",
             "anno_path": "Subway/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Walking", "path": "Walking/img", "startFrame": 1, "endFrame": 412, "nz": 4, "ext": "jpg",
             "anno_path": "Walking/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Walking2", "path": "Walking2/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg",
             "anno_path": "Walking2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Woman", "path": "Woman/img", "startFrame": 1, "endFrame": 597, "nz": 4, "ext": "jpg",
             "anno_path": "Woman/groundtruth_rect.txt",
             "object_class": "person"}
        ]

        return sequence_info_list
