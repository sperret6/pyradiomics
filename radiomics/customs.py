from radiomics import base
import SimpleITK as sitk
import numpy as np
import os
import itertools
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, euclidean

class RadiomicsCustoms(base.RadiomicsFeaturesBase):
    """
    Feature class docstring
    """

    def __init__(self, inputImage, inputMask, **kwargs):
        super(RadiomicsCustoms, self).__init__(inputImage, inputMask, **kwargs)
        self.cc_mask = sitk.ConnectedComponent(inputMask)
        self.contour_mask = sitk.SimpleContourExtractor(inputMask)
        self.cc_contour_mask = sitk.ConnectedComponent(self.contour_mask)

        database_path = "/media/sperret/11eaad1d-4fc4-47ac-951a-674eb3fe3b46/PSMA_Lea"
        sitk.WriteImage(self.cc_contour_mask, os.path.join(database_path, "patient_001" , "PSMA_PET", "PSMA_contour_001.nii.gz"))

    def getNRoiFeatureValue(self):
        return np.max(self.cc_mask)

    def getDmaxFeatureValue(self):

        coords = np.argwhere(sitk.GetArrayFromImage(self.inputMask))
        hull = ConvexHull(coords)
        hull_points = coords[hull.vertices]
        physical = [
          self.inputImage.TransformIndexToPhysicalPoint(tuple(int(x) for x in h))
          for h in hull_points
        ]
        distances = pdist(physical, metric='euclidean')
        return np.max(distances)

    def getMedPCDFeatureValue(self):

      stats = sitk.LabelShapeStatisticsImageFilter()
      stats.Execute(self.cc_mask)

      distances = []
      for label in stats.GetLabels():
        centroid = stats.GetCentroid(label)
        contour_lesion = sitk.GetArrayFromImage(self.cc_contour_mask)
        indices = np.argwhere(contour_lesion == label)
        physical_points = self._transformIndexToPysicalPoint(indices)
        if label == 1:
          for p in physical_points:

            distances.append(euclidean(p, centroid))
      return np.median(distances)


    def _transformIndexToPysicalPoint(self, list):

      list = np.flip(list)
      physical = [
        self.inputImage.TransformIndexToPhysicalPoint(tuple(int(x) for x in h))
        for h in list
      ]
      return physical


