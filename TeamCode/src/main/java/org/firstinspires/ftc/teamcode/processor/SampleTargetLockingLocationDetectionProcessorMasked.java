package org.firstinspires.ftc.teamcode.processor;

import android.graphics.Canvas;
import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class SampleTargetLockingLocationDetectionProcessorMasked implements VisionProcessor {

    Mat clrMat = new Mat();
    Mat maskedMat = new Mat();
    Mat rangedMat = new Mat();
    Mat thresholdMat = new Mat();
    Mat morphedMat = new Mat();
    Telemetry telemetry;

    Scalar redLow = new Scalar(0, 150, 75);
    Scalar redHigh = new Scalar(255, 255, 160);
    Scalar redOutlineLow = new Scalar(130, 175, 75);
    Scalar redOutlineHigh = new Scalar(255, 255, 255);
    Scalar blueLow = new Scalar(0, 0, 150);
    Scalar blueHigh = new Scalar(130, 255, 255);
    Scalar blueOutlineLow = new Scalar(0, 0, 150);
    Scalar blueOutlineHigh = new Scalar(255, 255, 255);
    Scalar yellowLow = new Scalar(130, 0, 0);
    Scalar yellowHigh = new Scalar(255, 255, 75);
    Scalar yellowOutlineLow = new Scalar(210, 0, 0);
    Scalar yellowOutlineHigh = new Scalar(255, 255, 255);

    Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
    Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
    Mat closeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10));

    static Scalar green = new Scalar(0, 255, 0);

    public SampleTargetLockingLocationDetectionProcessorMasked(Telemetry telemetry) {
        this.telemetry = telemetry;
    }

    @Override
    public void init(int width, int height, CameraCalibration calibration) {
    }

    @Override
    public Object processFrame(Mat input, long captureTimeNanos) {
        for(MatOfPoint contour : findContours(input, "YELLOW"))
        {
            if (Imgproc.contourArea(contour) > 3000)
                SeparateContours(contour, input);
        }

        return null;
    }

    private void analyzeContour(MatOfPoint contour, Mat input) {
        ArrayList<Point> contourArray = new ArrayList<Point>(contour.toList());
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

        RotatedRect rotatedRectFitToContour = Imgproc.minAreaRect(contour2f);
        for (Point point : contourArray) {

            Imgproc.circle(input, point, 1, green, 2);

        }
//        drawRotatedRect(rotatedRectFitToContour, input);
    }
    private void analyzeRightAng(MatOfPoint contour, Mat input) {
        Point[] contourArray = contour.toArray();
        ArrayList<Point> corners = new ArrayList<Point>();
        double angle;
        int corner1 = 0;
        int corner2;
        for (int i = 0; i < contourArray.length; i++) {
            for (int j = corner1 + 1; j < i; j++) {
                corner2 = j;
                angle = InverseLawOfCosSSS(
                        CalcDistance(contourArray[corner1], contourArray[corner2]),
                        CalcDistance(contourArray[i], contourArray[corner2]),
                        CalcDistance(contourArray[i], contourArray[corner1]));

                if (Math.abs(angle - 90) < 10) {
                    corners.add(contourArray[corner2]);
                }
            }
        }
        for (Point point : corners) {
            Imgproc.circle(input, point, 1, green, 2);

        }
    }
    private void SeparateContours(MatOfPoint contour, Mat input) {
        MatOfPoint OriginalContour = new MatOfPoint();
        MatOfPoint SeparatedContour = new MatOfPoint();
        ArrayList<Point> OriginalContourArray = new ArrayList<Point>(contour.toList());
        ArrayList<Point> SeparatedContourArray = new ArrayList<Point>();
        double angle;
        double angle2;
        int corner = 0;
        int offset = 0;
        for (int i = offset; i < OriginalContourArray.size(); i += 10) {
            angle = Math.atan(Math.abs(OriginalContourArray.get(i).y - OriginalContourArray.get(corner).y) / Math.abs(OriginalContourArray.get(i).x - OriginalContourArray.get(corner).x));
            for (int j = corner; j < i; j++) {
                angle2 = Math.atan(Math.abs(OriginalContourArray.get(j).y - OriginalContourArray.get(corner).y) / Math.abs(OriginalContourArray.get(j).x - OriginalContourArray.get(corner).x));
            }
        }
        OriginalContour.fromList(OriginalContourArray);
        SeparatedContour.fromList(SeparatedContourArray);
//        analyzeContour(OriginalContour, input);
        analyzeContour(SeparatedContour, input);
    }

    private void SeparateContoursAdv(MatOfPoint contour, Mat input) {
        MatOfPoint OriginalContour = new MatOfPoint();
        MatOfPoint SeparatedContour = new MatOfPoint();
        ArrayList<Point> OriginalContourArray = new ArrayList<Point>(contour.toList());
        ArrayList<Point> SeparatedContourArray = new ArrayList<Point>();
        double angle;
        double angle2;
        int corner = 0;
        int offset = 0;
        for (int i = offset; i < OriginalContourArray.size(); i += 10) {
            angle = Math.atan(Math.abs(OriginalContourArray.get(i).y - OriginalContourArray.get(corner).y) / Math.abs(OriginalContourArray.get(i).x - OriginalContourArray.get(corner).x));
            for (int j = corner; j < i; j++) {
                angle2 = Math.atan(Math.abs(OriginalContourArray.get(j).y - OriginalContourArray.get(corner).y) / Math.abs(OriginalContourArray.get(j).x - OriginalContourArray.get(corner).x));
                if (angle2 > angle) {
                    SeparatedContourArray.add(OriginalContourArray.get(j));
                    OriginalContourArray.remove(j);
                    corner = j;
                    offset = corner;
                    break;
                }
            }
        }
        OriginalContour.fromList(OriginalContourArray);
        SeparatedContour.fromList(SeparatedContourArray);
//        analyzeContour(OriginalContour, input);
        analyzeContour(SeparatedContour, input);
    }

    private void analyzeContourWithDeriv(MatOfPoint contour, Mat input) {
        MatOfPoint OriginalContour = new MatOfPoint();
        ArrayList<Point> OriginalContourArray = new ArrayList<Point>();
        MatOfPoint D1 = TakeDerivOfContour(OriginalContour);
        MatOfPoint D2 = TakeDerivOfContour(D1);


    }
    private MatOfPoint TakeDerivOfContour(MatOfPoint OriginalContour) {
        MatOfPoint contour = new MatOfPoint();
        MatOfPoint DofContour = new MatOfPoint();
        ArrayList<Point> ContourArray = new ArrayList<Point>(OriginalContour.toList());
        ArrayList<Point> DofContourArray = new ArrayList<Point>();
        for (int i = 1; i < ContourArray.size()-1; i++){
            DofContourArray.add(new Point(i, CalcSlope(ContourArray.get(i-1), ContourArray.get(i+1))));
        }
        DofContour.fromList(DofContourArray);

        return DofContour;
    }
    private double CalcSlope (Point Point1, Point Point2) {
        return (Point2.y - Point1.y) / (Point2.x - Point1.x);
    }
    private double CalcDistance (Point Point1, Point Point2) {
        return Math.sqrt(Math.pow((Point1.x - Point2.x), 2) + Math.pow((Point1.y - Point2.y), 2));
    }
    private double InverseLawOfCosSSS (double a, double b, double c) {
        return Math.toDegrees(Math.acos((Math.pow(a, 2) + Math.pow(b, 2) - Math.pow(c, 2))/(2*a*b)));
    }
    ArrayList<MatOfPoint> findContours(Mat input, String color) {
        ArrayList<MatOfPoint> contoursList = new ArrayList<>();

        Scalar lowYCrCb = new Scalar(0, 0, 0);
        Scalar highYCrCb = new Scalar(255, 255, 255);
        Scalar lowOutlineYCrCb = new Scalar(0, 0, 0);
        Scalar highOutlineYCrCb = new Scalar(255, 255, 255);

        if (color.equals("RED")) {
            lowYCrCb = redLow;
            highYCrCb = redHigh;
            lowOutlineYCrCb = redOutlineLow;
            highOutlineYCrCb = redOutlineHigh;
        }

        if (color.equals("BLUE")) {
            lowYCrCb = blueLow;
            highYCrCb = blueHigh;
            lowOutlineYCrCb = blueOutlineLow;
            highOutlineYCrCb = blueOutlineHigh;
        }

        if (color.equals("YELLOW")) {
            lowYCrCb = yellowLow;
            highYCrCb = yellowHigh;
            lowOutlineYCrCb = yellowOutlineLow;
            highOutlineYCrCb = yellowOutlineHigh;
        }
        Imgproc.cvtColor(input, clrMat, Imgproc.COLOR_RGB2YCrCb);
        Core.inRange(clrMat, lowYCrCb, highYCrCb, rangedMat);
        maskedMat.release();
        Core.bitwise_and(input, input, maskedMat, rangedMat);
        maskedMat.copyTo(input);
        Imgproc.cvtColor(input, maskedMat, Imgproc.COLOR_RGB2YCrCb);
        Core.inRange(maskedMat, lowYCrCb, highYCrCb, rangedMat);
        Imgproc.threshold(rangedMat, thresholdMat, 0, 255, Imgproc.THRESH_BINARY);
        morphMask(thresholdMat, morphedMat);
//        morphedMat.copyTo(input);
        Imgproc.findContours(morphedMat, contoursList, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        return contoursList;
    }

    void morphMask(Mat input, Mat output)
    {
        Imgproc.erode(input, output, erodeElement);
        Imgproc.erode(output, output, erodeElement);

        Imgproc.dilate(output, output, dilateElement);
        Imgproc.dilate(output, output, dilateElement);

        Imgproc.morphologyEx(input, output, Imgproc.MORPH_CLOSE, closeElement);
        Imgproc.morphologyEx(output, output, Imgproc.MORPH_CLOSE, closeElement);


    }


    static void drawRotatedRect(RotatedRect rect, Mat drawOn)
    {
        Point[] points = new Point[4];
        rect.points(points);

        for(int i = 0; i < 4; ++i)
        {
            Imgproc.line(drawOn, points[i], points[(i+1)%4], green, 2);
        }

//        drawMidline(rect, points, drawOn);

        if (rect.size.width < rect.size.height) {
            rect.angle += 90;
        }

        double angle = 180 - (rect.angle);
        drawTagText(rect, Integer.toString((int) Math.round(angle)) + " deg", 100, 50, drawOn);
        drawTagText(rect, Integer.toString((int) Math.round(rect.size.area())), 100, 100, drawOn);
    }

    static void drawTagText(RotatedRect rect, String text, double xOffset, double yOffset, Mat mat)
    {
        Imgproc.putText(
                mat, // The buffer we're drawing on
                text, // The text we're drawing
                new Point( // The anchor point for the text
                        rect.center.x+xOffset,  // x anchor point
                        rect.center.y+yOffset), // y anchor point
                Imgproc.FONT_HERSHEY_PLAIN, // Font
                1, // Font size
                green, // Font color
                2); // Font thickness
    }

    static void drawMidline(RotatedRect rect, Point[] points, Mat drawOn)
    {

        double heightToWidthRatio = (double) Math.round((rect.size.height/rect.size.width) * 100) / 100;
        int ind = rect.size.height < rect.size.width ? 0 : 1;
        Point midpt1 = new Point((points[ind%4].x + points[(ind+1)%4].x)/2, (points[ind%4].y + points[(ind+1)%4].y)/2);
        Point midpt2 = new Point((points[(ind+2)%4].x + points[(ind+3)%4].x)/2, (points[(ind+2)%4].y + points[(ind+3)%4].y)/2);
        if (0.8 < heightToWidthRatio && heightToWidthRatio < 1.8)
        {
            Imgproc.line(drawOn, midpt1, midpt2, green, 2);
        }
        drawTagText(rect, Double.toString(heightToWidthRatio), 100, 75, drawOn);
    }

    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight, float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {
    }


}

