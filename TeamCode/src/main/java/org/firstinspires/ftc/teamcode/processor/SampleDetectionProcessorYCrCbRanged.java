package org.firstinspires.ftc.teamcode.processor;

import android.graphics.Canvas;
import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

public class SampleDetectionProcessorYCrCbRanged implements VisionProcessor {

    Mat clrMat = new Mat();
    Mat maskedMat = new Mat();
    Mat rangedMat = new Mat();
    Mat thresholdMat = new Mat();
    Mat morphedMat = new Mat();

    Scalar redLow = new Scalar(0, 150, 75);
    Scalar redHigh = new Scalar(255, 255, 160);
    Scalar blueLow = new Scalar(0, 0, 150);
    Scalar blueHigh = new Scalar(130, 255, 255);
    Scalar yellowLow = new Scalar(130, 0, 0);
    Scalar yellowHigh = new Scalar(255, 255, 75);

    Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
    Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6, 6));

    static Scalar green = new Scalar(0, 255, 0);

    @Override
    public void init(int width, int height, CameraCalibration calibration) {
    }

    @Override
    public Object processFrame(Mat input, long captureTimeNanos) {

        for(MatOfPoint contour : findContours(input, "RED"))
        {
            analyzeContour(contour, input);
        }

        return null;
    }

    private void analyzeContour(MatOfPoint contour, Mat input) {

        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

        RotatedRect rotatedRectFitToContour = Imgproc.minAreaRect(contour2f);
        if (rotatedRectFitToContour.size.area()>3000) {
            drawRotatedRect(rotatedRectFitToContour, input);
        }
    }

    ArrayList<MatOfPoint> findContours(Mat input, String color)
    {
        ArrayList<MatOfPoint> contoursList = new ArrayList<>();

        Scalar lowYCrCb = new Scalar(0, 0, 0);
        Scalar highYCrCb = new Scalar(255, 255, 255);

        if (color.equals("RED")) {
            lowYCrCb = redLow;
            highYCrCb = redHigh;
        }

        if (color.equals("BLUE")) {
            lowYCrCb = blueLow;
            highYCrCb = blueHigh;
        }

        if (color.equals("YELLOW")) {
            lowYCrCb = yellowLow;
            highYCrCb = yellowHigh;
        }
        Imgproc.cvtColor(input, clrMat, Imgproc.COLOR_RGB2YCrCb);
        Core.inRange(clrMat, lowYCrCb, highYCrCb, rangedMat);
        Imgproc.threshold(rangedMat, thresholdMat, 0, 255, Imgproc.THRESH_BINARY);
        morphMask(thresholdMat, morphedMat);
        Imgproc.findContours(morphedMat, contoursList, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        return contoursList;
    }

    void morphMask(Mat input, Mat output)
    {
        Imgproc.erode(input, output, erodeElement);
        Imgproc.erode(output, output, erodeElement);

        Imgproc.dilate(output, output, dilateElement);
        Imgproc.dilate(output, output, dilateElement);
    }


    static void drawRotatedRect(RotatedRect rect, Mat drawOn)
    {
        Point[] points = new Point[4];
        rect.points(points);

        for(int i = 0; i < 4; ++i)
        {
            Imgproc.line(drawOn, points[i], points[(i+1)%4], green, 2);
        }

        double heightToWidthRatio = (double) Math.round((rect.size.height/rect.size.width) * 100) / 100;
        int ind = rect.size.height < rect.size.width ? 0 : 1;
        Point midpt1 = new Point((points[ind%4].x + points[(ind+1)%4].x)/2, (points[ind%4].y + points[(ind+1)%4].y)/2);
        Point midpt2 = new Point((points[(ind+2)%4].x + points[(ind+3)%4].x)/2, (points[(ind+2)%4].y + points[(ind+3)%4].y)/2);
        if (0.8 < heightToWidthRatio && heightToWidthRatio < 1.8)
        {
            Imgproc.line(drawOn, midpt1, midpt2, green, 2);
        }
        if (rect.size.width < rect.size.height) {
            rect.angle += 90;
        }

        double angle = 180 - (rect.angle);
        drawTagText(rect, Integer.toString((int) Math.round(angle)) + " deg", 100, 50, drawOn);
        drawTagText(rect, Integer.toString((int) Math.round(rect.size.area())), 100, 100, drawOn);
        drawTagText(rect, Double.toString(heightToWidthRatio), 100, 75, drawOn);
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

    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight, float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {
    }


}

