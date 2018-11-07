


# QC types
ERROR = "error"
WARNING = "warning"

# Explanation
OUT_OF_RANGE= "out of range value"
def range_check(value):
    if value <0 or value>1000:
        return {"QC":ERROR, "explain":OUT_OF_RANGE, "value":value}
