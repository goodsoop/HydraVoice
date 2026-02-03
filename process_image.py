import numpy as np
import cv2 as cv


def identify_circle(image, imgContour, x, y):

    circles = cv.HoughCircles(
        image,
        cv.HOUGH_GRADIENT,
        1,
        100,
        param1=50,
        param2=30,
        minRadius=200,
        maxRadius=600,
    )

    if circles is not None:
        cv.putText(
            imgContour,
            f"Circle",
            (x, y - 5),
            cv.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    else:

        cv.putText(
            imgContour,
            f"Unsure",
            (x, y - 5),
            cv.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


def empty(a):
    pass


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(
                        imgArray[x][y], (0, 0), None, scale, scale
                    )
                else:
                    imgArray[x][y] = cv.resize(
                        imgArray[x][y],
                        (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                        None,
                        scale,
                        scale,
                    )
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0]), None, scale, scale
                )
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img, imgContour, contourMode, gray, original):
    contours, hierarchy = cv.findContours(img, contourMode, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1000:
            shape = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
            # x_coordinate = shape.ravel()[0]
            # y_coordinate = shape.ravel()[1] - 15
            # print(len(shape))
            # if len(shape) > 13:
            rows = gray.shape[0]
            circles = cv.HoughCircles(
                gray,
                cv.HOUGH_GRADIENT,
                1,
                40,
                param1=50,
                param2=30,
                minRadius=144,
                maxRadius=600,
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                x, y, r = circles[0][0]
                cv.circle(imgContour, (x, y), r, (0, 255, 0), 2)  # Circle outline

            # approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            # # print(len(approx))
            # x, y, w, h = cv.boundingRect(approx)
            # cv.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # identify_circle(gray, imgContour, x, y)


cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 640, 240)
cv.createTrackbar("Threshold1", "Parameters", 152, 255, empty)
cv.createTrackbar("Threshold2", "Parameters", 170, 255, empty)
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    imgBlur = cv.medianBlur(frame, 3)
    imgContour = frame.copy()
    imgContour2 = frame.copy()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    threshold1 = cv.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv.getTrackbarPos("Threshold2", "Parameters")
    gray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)
    imgCanny = cv.Canny(gray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDil, imgContour, cv.RETR_EXTERNAL, gray, frame)
    imgStack = stackImages(
        0.6, ([frame, imgBlur, imgCanny], [imgDil, imgContour, imgContour])
    )
    cv.imshow("Result", imgStack)

    if cv.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
