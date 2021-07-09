
import imutils
import cv2


method = "csrt"

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create
}

trackers = cv2.legacy.MultiTracker_create()
vs = cv2.VideoCapture('0_1_mini.mp4')

fps = None

# loop throw video
while True:
    ret, frame = vs.read()
    if ret == True:

        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

    else:
        break

    (success, boxes) = trackers.update(frame)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # if success:
    #     (x, y, w, h) = [int(v) for v in box]
    #     cv2.rectangle(frame, (x, y), (x + w, y + h),
    #                   (0, 255, 0), 2)
    info = [
        ("Tracker",method),
        ("Success", "Yes" if success else "No"),
    ]

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # select object
    if key == ord("s"):
        box = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)
        tracker = OPENCV_OBJECT_TRACKERS[method]()
        trackers.add(tracker, frame, box)

    elif key == ord("q"):
        break
vs.release()

cv2.destroyAllWindows()
