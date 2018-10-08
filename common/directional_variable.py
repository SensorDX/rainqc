import numpy as np



def average2(angles):
    """

       Args:
           angles: np.array or list of angles.

       Returns: average of the angles.

       """
    if angles is None:
        return ValueError("Empty vectors")
    if type(angles) is list:
        angles = np.array(angles)

    sin = np.sum(np.sin(angles * np.pi / 180))
    cos = np.sum(np.cos(angles * np.pi / 180))

    avg_angle = np.arctan2(sin, cos)*180/np.pi
    if sin > 0 and cos > 0:
        return avg_angle
    elif cos < 0:
        return avg_angle + 180
    else:
        return avg_angle + 360
    #return avg_angle

if __name__ == '__main__':
    print average2([355, 5, 15, 20])
    print average2([0, 360, 180])
