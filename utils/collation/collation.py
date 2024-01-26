import numpy as np
import torch
import MinkowskiEngine as ME


class CollateFN:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_d = []
        list_idx = []
        list_xyz = []
        list_sampled = []

        for d in list_data:
            # coordinates
            # features
            # sem_labels
            # center_labels
            # idx
            list_d.append((d["coordinates"].to(self.device), d["features"].to(self.device), d["sem_labels"]))
            list_xyz.append(d["xyz"])
            list_idx.append(d["idx"].view(-1, 1))
            list_sampled.append(d["sampled_idx"].view(-1))

        coordinates_batch, features_batch, sem_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_d)

        idx_batch = torch.cat(list_idx, dim=0)
        sampled_idx_batch = torch.cat(list_sampled, dim=0)
        xyz_batch = torch.cat(list_xyz, dim=0)

        return {"coordinates": coordinates_batch,
                "xyz": xyz_batch,
                "features": features_batch,
                "sem_labels": sem_labels_batch,
                "idx": idx_batch,
                "sampled_idx": sampled_idx_batch}


class CollateFNSingleSource:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_d = []
        list_idx = []
        list_xyz = []
        list_sampled = []

        for d in list_data:
            # coordinates
            # features
            # sem_labels
            # center_labels
            # idx
            list_d.append((d["coordinates"].to(self.device), d["features"].to(self.device), d["sem_labels"]))
            list_xyz.append(d["xyz"])
            list_idx.append(d["idx"].view(-1, 1))
            list_sampled.append(d["sampled_idx"].view(-1))

        coordinates_batch, features_batch, sem_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_d)

        idx_batch = torch.cat(list_idx, dim=0)
        sampled_idx_batch = torch.cat(list_sampled, dim=0)
        xyz_batch = torch.cat(list_xyz, dim=0)

        return {"source_coordinates0": coordinates_batch,
                "source_xyz0": xyz_batch,
                "source_features0": features_batch,
                "source_sem_labels0": sem_labels_batch,
                "source_idx0": idx_batch,
                "source_sampled_idx0": sampled_idx_batch}


class CollateFNEval:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_d = []
        list_xyz = []
        list_idx = []
        list_ins = []
        list_center = []
        list_sampled = []
        list_inverse = []
        list_inverse_idx = []

        counter = 0
        for d in list_data:
            # coordinates
            # features
            # sem_labels
            # ins_labels
            # center_labels
            # idx
            list_d.append((d["coordinates"].to(self.device), d["features"].to(self.device), d["sem_labels"]))
            list_xyz.append(d["xyz"])
            list_ins.append((d["ins_labels"]))
            list_center.append((d['center_labels']))
            list_idx.append(d["idx"].view(-1, 1))
            list_sampled.append(d["sampled_idx"].view(-1))
            list_inverse.append(d["inverse_map"].view(-1, 1))
            list_inverse_idx.append(torch.ones_like(d["inverse_map"].view(-1, 1), dtype=torch.uint8) * counter)
            counter += 1

        coordinates_batch, features_batch, sem_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_d)
        xyz_batch = torch.cat(list_xyz, dim=0)
        ins_labels_batch = torch.cat(list_ins, dim=0)
        center_labels_batch = torch.cat(list_center, dim=0)
        idx_batch = torch.cat(list_idx, dim=0)
        sampled_idx_batch = torch.cat(list_sampled, dim=0)
        list_inverse = torch.cat(list_inverse, dim=0)
        list_inverse_idx = torch.cat(list_inverse_idx, dim=0)
        list_inverse = torch.cat([list_inverse, list_inverse_idx], dim=-1)

        return {"coordinates": coordinates_batch,
                "xyz": xyz_batch,
                "features": features_batch,
                "sem_labels": sem_labels_batch,
                "ins_labels": ins_labels_batch,
                "center_labels": center_labels_batch,
                "idx": idx_batch,
                "sampled_idx": sampled_idx_batch,
                "inverse_idx": list_inverse}


class CollateFNMultiSource:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_d0 = []
        list_d1 = []
        list_idx0 = []
        list_idx1 = []
        # list_xyz0 = []
        # list_xyz1 = []
        # list_sampled0 = []
        # list_sampled1 = []

        for d in list_data:
            # source_coordinates0
            # source_coordinates1
            # source_xyz0
            # source_xyz1
            # source_features0
            # source_features1
            # source_sem_labels0
            # source_sem_labels1
            # source_sampled_idx0
            # source_idx0
            # source_idx1
            # source_inverse_map0
            # source_inverse_map1

            list_d0.append((d["source_coordinates0"].to(self.device), d["source_features0"].to(self.device), d["source_sem_labels0"]))
            list_d1.append((d["source_coordinates1"].to(self.device), d["source_features1"].to(self.device), d["source_sem_labels1"]))
            # list_xyz0.append(d["source_xyz0"])
            # list_xyz1.append(d["source_xyz1"])
            # list_sampled0.append(d["source_sampled_idx0"].view(-1))
            # list_sampled1.append(d["source_sampled_idx1"].view(-1))
            list_idx0.append(d["source_idx0"].view(-1))
            list_idx1.append(d["source_idx1"].view(-1))

        source_coordinates_batch0, source_features_batch0, source_sem_labels_batch0 = ME.utils.SparseCollation(dtype=torch.float32, device=self.device)(list_d0)
        source_coordinates_batch1, source_features_batch1, source_sem_labels_batch1 = ME.utils.SparseCollation(dtype=torch.float32, device=self.device)(list_d1)

        source_idx0_batch = torch.cat(list_idx0, dim=0)
        source_idx1_batch = torch.cat(list_idx1, dim=0)

        # source_sampled_idx_batch0 = torch.cat(list_sampled0, dim=0)
        # source_sampled_idx_batch1 = torch.cat(list_sampled1, dim=0)

        # source_xyz_batch0 = torch.cat(list_xyz0, dim=0)
        # source_xyz_batch1 = torch.cat(list_xyz1, dim=0)

        return {"source_coordinates0": source_coordinates_batch0,
                "source_coordinates1": source_coordinates_batch1,
                "source_features0": source_features_batch0,
                "source_features1": source_features_batch1,
                "source_sem_labels0": source_sem_labels_batch0,
                "source_sem_labels1": source_sem_labels_batch1,
                "source_idx0": source_idx0_batch,
                "source_idx1": source_idx1_batch}


class CollateFNSingleSourceBEV:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_d = []
        list_idx = []
        list_xyz = []
        list_sampled = []
        # list_bev = []
        list_bev_labels = []
        list_bev_selected_idx = []
        # list_quantized_bottle = []
        # list_mapping = []

        # b_idx = 0
        for d in list_data:
            # coordinates
            # features
            # sem_labels
            # center_labels
            # idx
            list_d.append((d["coordinates"].to(self.device), d["features"].to(self.device), d["sem_labels"]))
            list_xyz.append(d["xyz"])
            list_idx.append(d["idx"].view(-1, 1))
            list_sampled.append(d["sampled_idx"].view(-1))
            # list_bev.append(d["bev_image"].unsqueeze(0))
            list_bev_labels.append((d["bev_labels"].unsqueeze(0)))
            list_bev_selected_idx.append(d["bev_selected_idx"].unsqueeze(0))

            # # need to preserve batch indexes
            # quantized_bottle_coords = d["quantized_bottle_coords"]
            # b_idx_points = torch.ones(quantized_bottle_coords.shape[0], dtype=torch.float32).view(-1, 1) * b_idx
            # quantized_bottle_coords = torch.cat([b_idx_points, quantized_bottle_coords], dim=-1)
            # list_quantized_bottle.append(quantized_bottle_coords)
            # b_idx += 1

            # list_mapping.append(d["bottle_bev_mapping"])

        coordinates_batch, features_batch, sem_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_d)

        idx_batch = torch.cat(list_idx, dim=0)
        sampled_idx_batch = torch.cat(list_sampled, dim=0)
        xyz_batch = torch.cat(list_xyz, dim=0)
        # bev_batch = torch.cat(list_bev, dim=0)
        bev_batch_labels = torch.cat(list_bev_labels, dim=0)
        bev_batch_selected_idx = torch.cat(list_bev_selected_idx, dim=0)
        # quantized_bottle_batch = torch.cat(list_quantized_bottle, dim=0)
        # mapping_batch = torch.cat(list_mapping, dim=0)

        return {"source_coordinates0": coordinates_batch,
                "source_xyz0": xyz_batch,
                "source_features0": features_batch,
                "source_sem_labels0": sem_labels_batch,
                "source_idx0": idx_batch,
                "source_sampled_idx0": sampled_idx_batch,
                # "source_bev_map0": bev_batch,
                "source_bev_labels0": bev_batch_labels,
                "source_bev_sampled_idx0": bev_batch_selected_idx}


class CollateFNSingleSourceBEVMultiLevel:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_d = []
        list_idx = []
        list_xyz = []
        list_sampled = []

        bev_keys = list_data[0]['bev_labels'].keys()
        list_bev_labels = {b: [] for b in bev_keys}
        list_bev_selected_idx = {b: [] for b in bev_keys}

        for d in list_data:
            # coordinates
            # features
            # sem_labels
            # center_labels
            # idx
            list_d.append((d["coordinates"].to(self.device), d["features"].to(self.device), d["sem_labels"]))
            list_xyz.append(d["xyz"])
            list_idx.append(d["idx"].view(-1, 1))
            list_sampled.append(d["sampled_idx"].view(-1))
            bev_labels_tmp = d["bev_labels"]
            bev_selected_tmp = d["bev_selected_idx"]

            for k in bev_keys:
                list_bev_labels[k].append(bev_labels_tmp[k].unsqueeze(0))
                list_bev_selected_idx[k].append(bev_selected_tmp[k].unsqueeze(0))

        coordinates_batch, features_batch, sem_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_d)

        idx_batch = torch.cat(list_idx, dim=0)
        sampled_idx_batch = torch.cat(list_sampled, dim=0)
        xyz_batch = torch.cat(list_xyz, dim=0)
        bev_batch_labels = {k: torch.cat(list_bev_labels[k], dim=0) for k in bev_keys}
        bev_batch_selected_idx = {k: torch.cat(list_bev_selected_idx[k], dim=0) for k in bev_keys}

        return {"source_coordinates0": coordinates_batch,
                "source_xyz0": xyz_batch,
                "source_features0": features_batch,
                "source_sem_labels0": sem_labels_batch,
                "source_idx0": idx_batch,
                "source_sampled_idx0": sampled_idx_batch,
                "source_bev_labels0": bev_batch_labels,
                "source_bev_sampled_idx0": bev_batch_selected_idx}


class CollateFNMultiSourceBEVMultiLevel:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries for BEV dataset
        """
        list_d0 = []
        list_idx0 = []
        list_xyz0 = []
        list_sampled0 = []
        list_d1 = []
        list_idx1 = []
        list_xyz1 = []
        list_sampled1 = []

        bev_keys0 = list_data[0]['bev_labels0'].keys()
        list_bev_labels0 = {b: [] for b in bev_keys0}
        list_bev_selected_idx0 = {b: [] for b in bev_keys0}

        bev_keys1 = list_data[0]['bev_labels1'].keys()
        list_bev_labels1 = {b: [] for b in bev_keys1}
        list_bev_selected_idx1 = {b: [] for b in bev_keys1}

        bev_keys = set(list(bev_keys0) + list(bev_keys1))

        for d in list_data:
            # coordinates
            # features
            # sem_labels
            # center_labels
            # idx
            list_d0.append((d["coordinates0"].to(self.device), d["features0"].to(self.device), d["sem_labels0"]))
            list_xyz0.append(d["xyz0"])
            list_idx0.append(d["idx0"].view(-1, 1))
            list_sampled0.append(d["sampled_idx0"].view(-1))
            bev_labels_tmp0 = d["bev_labels0"]
            bev_selected_tmp0 = d["bev_selected_idx0"]

            list_d1.append((d["coordinates1"].to(self.device), d["features1"].to(self.device), d["sem_labels1"]))
            list_xyz1.append(d["xyz1"])
            list_idx1.append(d["idx1"].view(-1, 1))
            list_sampled1.append(d["sampled_idx1"].view(-1))
            bev_labels_tmp1 = d["bev_labels1"]
            bev_selected_tmp1 = d["bev_selected_idx1"]

            for k in bev_keys:
                if k in bev_keys0:
                    list_bev_labels0[k].append(bev_labels_tmp0[k].unsqueeze(0))
                    list_bev_selected_idx0[k].append(bev_selected_tmp0[k].unsqueeze(0))
                if k in bev_keys1:
                    list_bev_labels1[k].append(bev_labels_tmp1[k].unsqueeze(0))
                    list_bev_selected_idx1[k].append(bev_selected_tmp1[k].unsqueeze(0))

        coordinates_batch0, features_batch0, sem_labels_batch0 = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_d0)

        coordinates_batch1, features_batch1, sem_labels_batch1 = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_d1)

        idx_batch0 = torch.cat(list_idx0, dim=0)
        sampled_idx_batch0 = torch.cat(list_sampled0, dim=0)
        xyz_batch0 = torch.cat(list_xyz0, dim=0)
        bev_batch_labels0 = {k: torch.cat(list_bev_labels0[k], dim=0) for k in bev_keys0}
        bev_batch_selected_idx0 = {k: torch.cat(list_bev_selected_idx0[k], dim=0) for k in bev_keys0}

        idx_batch1 = torch.cat(list_idx1, dim=0)
        sampled_idx_batch1 = torch.cat(list_sampled1, dim=0)
        xyz_batch1 = torch.cat(list_xyz1, dim=0)
        bev_batch_labels1 = {k: torch.cat(list_bev_labels1[k], dim=0) for k in bev_keys1}
        bev_batch_selected_idx1 = {k: torch.cat(list_bev_selected_idx1[k], dim=0) for k in bev_keys1}

        return {"source_coordinates0": coordinates_batch0,
                "source_xyz0": xyz_batch0,
                "source_features0": features_batch0,
                "source_sem_labels0": sem_labels_batch0,
                "source_idx0": idx_batch0,
                "source_sampled_idx0": sampled_idx_batch0,
                "source_bev_labels0": bev_batch_labels0,
                "source_bev_sampled_idx0": bev_batch_selected_idx0,
                "source_coordinates1": coordinates_batch1,
                "source_xyz1": xyz_batch1,
                "source_features1": features_batch1,
                "source_sem_labels1": sem_labels_batch1,
                "source_idx1": idx_batch1,
                "source_sampled_idx1": sampled_idx_batch1,
                "source_bev_labels1": bev_batch_labels1,
                "source_bev_sampled_idx1": bev_batch_selected_idx1}


class CollateFNBEVBounded:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        list_d = []
        list_idx = []
        list_xyz = []
        list_sampled = []

        bev_keys = list_data[0]['bev_labels'].keys()
        list_bev_labels = {b: [] for b in bev_keys}
        list_bev_selected_idx = {b: [] for b in bev_keys}
        list_bounds = []

        for d in list_data:
            # coordinates
            # features
            # sem_labels
            # center_labels
            # idx
            # bounds_coords
            # bounds_features
            # bounds_labels
            list_d.append((d["coordinates"].to(self.device), d["features"].to(self.device), d["sem_labels"]))
            list_xyz.append(d["xyz"])
            list_idx.append(d["idx"].view(-1, 1))
            list_sampled.append(d["sampled_idx"].view(-1))
            list_bounds.append((d["bounds_coords"].to(self.device), d["bounds_features"].to(self.device), d["bounds_labels"]))
            bev_labels_tmp = d["bev_labels"]
            bev_selected_tmp = d["bev_selected_idx"]

            for k in bev_keys:
                list_bev_labels[k].append(bev_labels_tmp[k].unsqueeze(0))
                list_bev_selected_idx[k].append(bev_selected_tmp[k].unsqueeze(0))

        coordinates_batch, features_batch, sem_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                   device=self.device)(list_d)
        bounds_coords_batch, bounds_features_batch, _ = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                 device=self.device)(list_bounds)

        idx_batch = torch.cat(list_idx, dim=0)
        sampled_idx_batch = torch.cat(list_sampled, dim=0)
        xyz_batch = torch.cat(list_xyz, dim=0)
        bev_batch_labels = {k: torch.cat(list_bev_labels[k], dim=0) for k in bev_keys}
        bev_batch_selected_idx = {k: torch.cat(list_bev_selected_idx[k], dim=0) for k in bev_keys}

        return {"source_coordinates0": coordinates_batch,
                "source_xyz0": xyz_batch,
                "source_features0": features_batch,
                "source_sem_labels0": sem_labels_batch,
                "source_idx0": idx_batch,
                "source_sampled_idx0": sampled_idx_batch,
                "source_bev_labels0": bev_batch_labels,
                "source_bev_sampled_idx0": bev_batch_selected_idx,
                "source_bounds_coordinates0": bounds_coords_batch,
                "source_bounds_features0": bounds_features_batch}


class CollateSTRLFN:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, list_data):
        r"""
        Collation function for MinkowskiEngine.SparseTensor that creates batched
        coordinates given a list of dictionaries.
        """
        """
        Input has the following keys:
            "coordinates"
            "xyz"
            "features"
            "sem_labels"
            "sampled_idx"
            "idx":
            "inverse_map"
            "next_coordinates"
            "next_xyz"
            "next_features"
            "next_sem_labels"
            "next_sampled_idx"
            "next_inverse_map"
        """

        list_d = []
        list_next = []
        list_idx = []
        list_xyz = []
        list_next_xyz = []
        list_sampled = []
        list_next_sampled = []

        for d in list_data:
            # coordinates
            # features
            # sem_labels
            # center_labels
            # idx
            list_d.append((d["coordinates"].to(self.device), d["features"].to(self.device), d["sem_labels"]))
            list_xyz.append(d["xyz"])
            list_idx.append(d["idx"].view(-1, 1))
            list_sampled.append(d["sampled_idx"].view(-1))

            list_next.append((d["next_coordinates"].to(self.device), d["next_features"].to(self.device), d["next_sem_labels"]))
            list_next_xyz.append(d["next_xyz"])
            list_idx.append(d["idx"].view(-1, 1))
            list_next_sampled.append(d["next_sampled_idx"].view(-1))

        coordinates_batch, features_batch, sem_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                       device=self.device)(list_d)
        next_coordinates_batch, next_features_batch, next_sem_labels_batch = ME.utils.SparseCollation(dtype=torch.float32,
                                                                                       device=self.device)(list_next)

        idx_batch = torch.cat(list_idx, dim=0)
        sampled_idx_batch = torch.cat(list_sampled, dim=0)
        xyz_batch = torch.cat(list_xyz, dim=0)

        next_sampled_idx_batch = torch.cat(list_next_sampled, dim=0)
        next_xyz_batch = torch.cat(list_next_xyz, dim=0)

        return {"coordinates": coordinates_batch,
                "xyz": xyz_batch,
                "features": features_batch,
                "sem_labels": sem_labels_batch,
                "idx": idx_batch,
                "sampled_idx": sampled_idx_batch,
                "next_coordinates": next_coordinates_batch,
                "next_xyz": next_xyz_batch,
                "next_features": next_features_batch,
                "next_sem_labels": next_sem_labels_batch,
                "next_sampled_idx": next_sampled_idx_batch}

