
class Classes:
    def __init__(self):
        self.ers_to_galar = {
                # Ulcer
                "c26": "ulcer", "c27": "ulcer",
                "g18": "ulcer", "g19": "ulcer",
                "g50": "ulcer", "g51": "ulcer", "g52": "ulcer",
                # Polyp
                "c22": "polyp", "g14": "polyp",
                "g31": "polyp", "g45": "polyp",
                "c23": "polyp",
                # Active bleeding
                "b01": "active_bleeding", "g19": "active_bleeding",
                "g51": "active_bleeding", "g05": "active_bleeding",
                "c02": "active_bleeding",
                # Blood
                "b": "blood", "b01": "blood",
                # Erythema
                "g22": "erythema", "g56": "erythema",
                # Erosion
                "g20": "erosion", "g54": "erosion", "g07": "erosion",
                "g21": "erosion",  # duodenopathy:hemorrhagic
                "g55": "erosion",
                # Angiectasia
                "c01": "angiectasia",
                # IBD (Crohn's / UC)
                "c08": "IBD", "c09": "IBD", "c10": "IBD",
                "c32": "IBD", "c33": "IBD", "c34": "IBD",
                "g07": "IBD", "c03":"IBD", "c04":"IBD", "c16": "IBD", "c11": "IBD", "c25": "IBD",
                # Foreign body
                "c14": "foreign_body", "g30": "foreign_body", "g42": "foreign_body","c20": "foreign_body", "g64": "foreign_body",
                "g28": "foreign_body",  # esophageal_diverticulum
                "g29": "foreign_body",  # esophageal_fistula
                "g40": "foreign_body",  # gastric_diverticulum
                "g41": "foreign_body",  # gastric_fistula
                "g13": "foreign_body",
                "q03": "foreign_body",
                "q04": "foreign_body",
                "q10": "foreign_body",
                # Esophagitis
                "g23": "esophagitis", "g63": "esophagitis", "g67": "esophagitis",
                # Varices
                "g36": "varices", "g53": "varices", "c15": "varices",
                # Hematin (no direct class, approximate)
                "g47": "other",
                "c07": "other",
                "c12": "other",
                "c17": "other",  # lipoma
                "c19": "other",  # melanosis
                "c21": "other",  # pneumatosis_coli
                "c24": "other",  # postoperative_appearance
                "c28": "other",  # stricture:inflammatory
                "c30": "other",  # stricture:postoperative
                "c31": "other",  # submucosal_tumor
                "g": "other",
                "g01": "other", "g02": "other", "g03": "other", "g04": "other",
                "g08": "other", "g09": "other", "g10": "other", "g12": "other",
                "g15": "other", "g16": "other", "g24": "other", "g25": "other",
                "g27": "other", "g32": "other", "g33": "other", "g35": "other",
                "g37": "other", "g38": "other", "g43": "other", "g46": "other",
                "g48": "other", "g57": "other", "g58": "other", "g59": "other",
                "g60": "other", "g61": "other", "g62": "other", "g65": "other",
                "g66": "other", "g68": "other", "g69": "other", "g70": "other",
                "q01": "other",
                "q02": "other",
                "q05": "other",
                "q07": "other",
                "q08": "other",
                "q09": "other",
                # Celiac
                "g06": "celiac",
                # Cancer
                "c05": "cancer", "g11": "cancer", "g26": "cancer",
                "g39": "cancer", "c29": "cancer", "g44": "cancer",
                "c06": "cancer", "c18": "cancer",
                "g17": "cancer",
                "g34": "cancer",
                "g49": "cancer",

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
            "foreign_body", "esophagitis", "varices",
            "celiac", "cancer", "other", "healthy"
        ]