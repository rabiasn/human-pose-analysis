def midpoint(a, b):
    return (
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2
    )


def torso_height(points):
    """
    Omuz merkezi ile kalça merkezi arasındaki dikey mesafe.
    Noktalar normalize edilmiş (x, y) koordinatlarıdır.
    """

    shoulder_center = midpoint(
        points["left_shoulder"],
        points["right_shoulder"]
    )

    hip_center = midpoint(
        points["left_hip"],
        points["right_hip"]
    )

    return abs(hip_center[1] - shoulder_center[1])


def arm_raised(points, side):
    """
    side: 'left' veya 'right'

    Görüntü koordinatlarında küçük y değeri daha yukarıdadır.
    """

    shoulder = points[f"{side}_shoulder"]
    elbow = points[f"{side}_elbow"]
    wrist = points[f"{side}_wrist"]

    torso = torso_height(points)

    if torso <= 0:
        return False

    wrist_above_shoulder = (
        wrist[1] < shoulder[1] - 0.10 * torso
    )

    elbow_position_ok = (
        elbow[1] < shoulder[1] + 0.15 * torso
    )

    return wrist_above_shoulder and elbow_position_ok


def leg_raised(points, side):
    """
    Tek karede ilgili bacağın diğer bacağa göre
    yukarı kaldırılmış durumda olup olmadığını tahmin eder.

    side: 'left' veya 'right'
    """

    ankle = points[f"{side}_ankle"]
    knee = points[f"{side}_knee"]

    other = "right" if side == "left" else "left"

    other_ankle = points[f"{other}_ankle"]
    other_knee = points[f"{other}_knee"]

    torso = torso_height(points)

    if torso <= 0:
        return False

    ankle_higher = (
        ankle[1] < other_ankle[1] - 0.15 * torso
    )

    knee_higher = (
        knee[1] < other_knee[1] - 0.10 * torso
    )

    return ankle_higher or knee_higher


def distance(a, b):
    return (
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    ) ** 0.5


def hand_near_mouth(points, side):
    """
    Bileğin yüz/ağız bölgesine yakın olup olmadığını
    tek kare üzerinden tahmin eder.
    """

    wrist = points[f"{side}_wrist"]
    nose = points["nose"]

    torso = torso_height(points)

    if torso <= 0:
        return False

    normalized_distance = distance(wrist, nose) / torso

    return normalized_distance < 0.55


def body_turn_3d(points):
    """
    3D omuz ve kalça derinliklerinden gövdenin yönünü tahmin eder.

    Beklenen nokta formatı:
    (x, y, z)

    Sonuç:
    "right"   -> kişi kendi sağına dönük
    "left"    -> kişi kendi soluna dönük
    "front"   -> belirgin dönüş yok
    """

    left_depth = (
        points["left_shoulder"][2] +
        points["left_hip"][2]
    ) / 2

    right_depth = (
        points["right_shoulder"][2] +
        points["right_hip"][2]
    ) / 2

    depth_difference = left_depth - right_depth

    threshold = 0.03

    # Küçük z = kameraya daha yakın
    if depth_difference < -threshold:
        return "right"

    if depth_difference > threshold:
        return "left"

    return "front"


def analyze_pose_2d(points):
    """
    Tek karedeki 2D pozdan temel durumları çıkarır.
    """

    return {
        "left_arm_raised": arm_raised(points, "left"),
        "right_arm_raised": arm_raised(points, "right"),
        "left_leg_raised": leg_raised(points, "left"),
        "right_leg_raised": leg_raised(points, "right"),
        "left_hand_near_mouth": hand_near_mouth(points, "left"),
        "right_hand_near_mouth": hand_near_mouth(points, "right"),
    }


def analyze_pose_3d(points):
    """
    Tek karedeki 3D pozdan gövde yönünü çıkarır.
    """

    return {
        "body_direction": body_turn_3d(points)
    }
