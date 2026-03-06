from Factory.Matrials.Heat_Treatment.hardened_part import HardenedPart
from Factory.Matrials.Heat_Treatment.pure_metal import PureMetal


class HeatTreatmentMaterials(PureMetal, HardenedPart):
    def __init__(self):
        PureMetal.__init__(self)
        HardenedPart.__init__(self)

    def load_pure_metal(self):
        return self._load_pure_metal()

    def unload_hardened_part(self):
        return self._unload_hardened_part()