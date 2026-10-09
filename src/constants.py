"""Physical sizes, marker IDs and marker placement shared by every script. Lengths in metres."""

CUBE_SIDE_MM = 57.0              # Rubik's cube
CUBE_SIDE = CUBE_SIDE_MM / 1000
CUBE_MARKER = 0.8 * CUBE_SIDE    # printed marker on the side faces (45.6 mm)
# Real setup (6 Oct): table marker ID 13, hand marker ID 0 (on carton). Sizes of the black squares as measured
# by the stereo head in the pilot (`python -m src.pilot_check static`, median over 297 frames, epipolar error
# < 1 px); the cube came out 56.9-57.2 mm from its face centres, so the stereo scale is right to ~0.5%.
# Check them with a ruler if you can.
CUBE_TOP_MARKER = 0.0420         # ID 1 is smaller than the side markers (reprinted at ~92%?)
TABLE_MARKER = 0.1451
WRIST_MARKER = 0.0428

ARUCO_DICT = "DICT_4X4_50"       # cube and hand markers
TABLE_DICT = ARUCO_DICT          # change if pilot_check finds marker 13 in another dictionary
TABLE_ID = 13
CUBE_TOP_ID = 1
CUBE_SIDE_IDS = (2, 3, 4, 5)
CUBE_IDS = (CUBE_TOP_ID, *CUBE_SIDE_IDS)
WRIST_ID = 0

MARKER_SIZE = {TABLE_ID: TABLE_MARKER, WRIST_ID: WRIST_MARKER, CUBE_TOP_ID: CUBE_TOP_MARKER,
               **{i: CUBE_MARKER for i in CUBE_SIDE_IDS}}
assert len({TABLE_ID, WRIST_ID, *CUBE_IDS}) == 2 + len(CUBE_IDS), "marker IDs must be distinct"

# Marker frame (OpenCV ArUco): origin at the marker centre, x to the right, y toward the
# marker's top edge, z out of the face.
# Cube frame: origin at the cube centre, z up through the top face (ID 1), x toward the ID 2 face.
# Seen from above, side IDs 2 -> 3 -> 4 -> 5 go counter-clockwise; each side marker's top edge
# points to the top face; ID 1's top edge points to the ID 4 face. (The plan said ID 2; the real cube
# has it toward ID 4, measured by stereo on 6 Oct: top edge . face-4 normal = +1.00.)
# Each entry: the marker's x, y, z axes in cube coordinates. Edit here if the cube is glued differently.
CUBE_MARKER_AXES = {
    1: ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    2: ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    3: ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    4: ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    5: ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
}

# Wrist (hand) marker: on the back of the hand, top edge (marker +y) toward the fingertips.
# Hand yaw = heading of marker +y projected on the table plane.

CHECKER_INNER = (9, 6)           # inner corners per row, per column
CHECKER_SQUARE = 0.025
