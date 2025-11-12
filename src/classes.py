
class Classes:
    def __init__(self, binary: bool = False):
        self.binary = binary
        if binary:
            self.unified_classes = ["unhealthy", "healthy"]

            # Everything not healthy becomes unhealthy
            self.ers_to_galar = {
                **{code: "unhealthy" for code in [
                    "c26", "c27", "g18", "g19", "g50", "g51", "g52", "g10", "g33", "g60",  # ulcer
                    "c22", "g14", "g31", "g45", "c23",  # polyp
                    "b01", "g19", "g51", "g05", "c02", "g15", "g46", "g62",  # active bleeding
                    "b", "b01",  # blood
                    "g22", "g56", "c19", "c24", "c30", "g32", "g59",  # erythema
                    "g20", "g54", "g07", "g21", "g55", "g13", "g25", "g29", "g58", "g41", "g43",  # erosion
                    "c01", "g08", "c28",  # angiectasia
                    "c08", "c09", "c10", "c32", "c33", "c34", "g07", "c03", "c16", "c11", "c25",  # IBD
                    "c14", "g30", "g42", "c20", "g64", "g28", "g40", "g37",  # foreign body
                    "g23", "g63", "g67", "g02", "g04",  # esophagitis
                    "g36", "g53", "c15", "g65",  # varices
                    "g47", "c07", "c12", "c21", "g01", "g03", "g10", "g12", "g24", "g27",
                    "g57", "g61", "g66", "g68", "g69", # other
                    "g06",  # celiac
                    "c05", "g11", "g26", "g39", "c29", "g44", "c06", "c18", "g17", "g34", "g49",
                    "c04", "c17", "c31", "g09", "g16", "g35", "g38", "g48", "g70"  # cancer
                ]},
                **{code: "healthy" for code in ["h01", "h02", "h03", "h04", "h05", "h06", "h07"]}
            }
        else:
            self.ers_to_galar = {
                    # Ulcer
                    "c26": "ulcer", "c27": "ulcer",
                    "g18": "ulcer", "g19": "ulcer",
                    "g50": "ulcer", "g51": "ulcer", "g52": "ulcer",
                    "g10": "ulcer", "g33": "ulcer",
                    "g60": "ulcer",
                    # Polyp
                    "c22": "polyp", "g14": "polyp",
                    "g31": "polyp", "g45": "polyp",
                    "c23": "polyp",
                    # Active bleeding
                    "b01": "active_bleeding", "g19": "active_bleeding",
                    "g51": "active_bleeding", "g05": "active_bleeding",
                    "c02": "active_bleeding", "g15": "active_bleeding", 
                    "g46": "active_bleeding", "g62": "active_bleeding",
                    # Blood
                    "b": "blood", "b01": "blood",
                    # Erythema
                    "g22": "erythema", "g56": "erythema", "c19": "erythema",
                    "c24": "erythema", "c30": "erythema", "g32": "erythema",
                    "g59": "erythema",
                    # Erosion
                    "g20": "erosion", "g54": "erosion", "g07": "erosion",
                    "g21": "erosion",  # duodenopathy:hemorrhagic
                    "g55": "erosion", "g13": "erosion",
                    "g25": "erosion", "g29": "erosion",
                    "g58": "erosion", "g41": "erosion", "g43": "erosion",
                    # Angiectasia
                    "c01": "angiectasia", "g08": "angiectasia", "c28": "angiectasia",
                    # IBD (Crohn's / UC)
                    "c08": "IBD", "c09": "IBD", "c10": "IBD",
                    "c32": "IBD", "c33": "IBD", "c34": "IBD",
                    "g07": "IBD", "c03":"IBD", "c16": "IBD", "c11": "IBD", "c25": "IBD",
                    # Foreign body
                    "c14": "foreign_body", "g30": "foreign_body", "g42": "foreign_body","c20": "foreign_body", "g64": "foreign_body",
                    "g28": "foreign_body",  # esophageal_diverticulum
                    "g40": "foreign_body",  # gastric_diverticulum
                    "g37": "foreign_body",
                    # Esophagitis
                    "g23": "esophagitis", "g63": "esophagitis", "g67": "esophagitis", 
                    "g02": "esophagitis", "g04": "esophagitis",
                    # Varices
                    "g36": "varices", "g53": "varices", "c15": "varices", "g65": "varices",
                    # Hematin (no direct class, approximate)
                    "g47": "other",
                    "c07": "other",
                    "c12": "other",
                    "c21": "other",  # pneumatosis_coli
                    "g": "other",
                    "g01": "other", "g03": "other",
                    "g10": "other", "g12": "other",
                    "g24": "other",
                    "g27": "other",
                    "g57": "other",
                    "g61": "other",
                    "g66": "other", "g68": "other", "g69": "other",
                    # Celiac
                    "g06": "celiac",
                    # Cancer
                    "c05": "cancer", "g11": "cancer", "g26": "cancer",
                    "g39": "cancer", "c29": "cancer", "g44": "cancer",
                    "c06": "cancer", "c18": "cancer",
                    "g17": "cancer",
                    "g34": "cancer",
                    "g49": "cancer",
                    "c04": "cancer",
                    "c17": "cancer",
                    "c31": "cancer",
                    "g09": "cancer",
                    "g16": "cancer",
                    "g35": "cancer",
                    "g38": "cancer",
                    "g48": "cancer",
                    "g70": "cancer",

                    #healthy
                    "h01": "healthy",
                    "h02":"healthy",
                    "h03": "healthy",
                    "h04": "healthy",
                    "h05": "healthy",
                    "h06": "healthy",
                    "h07": "healthy",
                }
            
            self.unified_classes = [
                "ulcer", "polyp", "active_bleeding", "blood",
                "erythema", "erosion", "angiectasia", "IBD",
                "foreign_body", "esophagitis", "varices", "hematin",
                "celiac", "cancer", "lymphangioectasis", "other", "healthy"
            ]

        self.num_classes = len(self.unified_classes)