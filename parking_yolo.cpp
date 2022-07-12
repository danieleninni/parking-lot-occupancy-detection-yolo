/*
Parking Lot Occupancy Detection using YOLOv5
@author Daniele Ninni
*/

#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

// path to the YOLOv5 model in the ONNX format
const string YOLO_PATH = "YOLOv5/yolov5s.onnx";

// trackbars name
const string CONFIDENCE_THRESHOLD_TRACKBAR_NAME      = "confidence %";
const string OVERLAP_THRESHOLD_TRACKBAR_NAME         = "overlap %";
const string DETECTION_AREA_THRESHOLD_TRACKBAR_NAME  = "detection area";

// maximal position of the trackbars slider (the minimal position is always 0)
const int TRACKBAR_MAX = 100;

// original image size
const float ORIGINAL_IMAGE_WIDTH  = 2592.0;
const float ORIGINAL_IMAGE_HEIGHT = 1944.0;

// downsampled image size
const float DOWNSAMPLED_IMAGE_WIDTH  = 1000.0;
const float DOWNSAMPLED_IMAGE_HEIGHT = 750.0;

// BLOB size
const float BLOB_WIDTH  = 640.0;
const float BLOB_HEIGHT = 640.0;

// colors
const Scalar RED    = Scalar(0, 0, 255);
const Scalar GREEN  = Scalar(0, 255, 0);
const Scalar YELLOW = Scalar(0, 255, 255);

// data structure "Detection"
struct Detection
{
    float confidence{};
    Rect bounding_box;
};

// data structure "ParkingLot"
struct ParkingLot
{
    string slot_id;
    Rect bounding_box;
};

// function that loads the parking lots
void load_parking_lots(vector<ParkingLot>& parking_lots, string PARKING_LOTS_PATH, float PARKING_LOT_X_FACTOR, float PARKING_LOT_Y_FACTOR)
{
    // pointer to the file containing the parking lots
    fstream file(PARKING_LOTS_PATH, ios::in);

    // read the data from the file as string vector
    vector<string> row;
    string line, word, header;

    if (file.is_open()) {

        // skip the header
        getline(file, header);

        // read each entire row and store it in the string "line"
        while (getline(file, line)) {

            // clear the vector "row"
            row.clear();

            // break words
            stringstream s(line);

            // read each column of a row and store it in the string "word"
            while (getline(s, word, ',')) {

                // add each column of a row to the vector "row"
                row.push_back(word);

            }

            // slot ID
            string slot_id = row[0];

            // original coordinates of the loaded bounding box
            float X = stof(row[1]);
            float Y = stof(row[2]);
            float W = stof(row[3]);
            float H = stof(row[4]);

            // downsampled coordinates of the loaded bounding box
            int x = int(X * PARKING_LOT_X_FACTOR);
            int y = int(Y * PARKING_LOT_Y_FACTOR);
            int w = int(W * PARKING_LOT_X_FACTOR);
            int h = int(H * PARKING_LOT_Y_FACTOR);

            // store the parking lot in the vector "parking_lots"
            ParkingLot parking_lot;
            parking_lot.slot_id      = slot_id;
            parking_lot.bounding_box = Rect(x, y, w, h);
            parking_lots.push_back(parking_lot);

        }

    }
    else {

        cerr << "Error: could not open '" << PARKING_LOTS_PATH << "'!\n\n";

    }
}

// function that performs object detection using YOLOv5
void yolo_detect(Mat& blob, vector<Detection>& detections, Net& yolo, float CONFIDENCE_THRESHOLD, float BLOB_X_FACTOR, float BLOB_Y_FACTOR)
{
    // set the input value for YOLOv5
    yolo.setInput(blob);

    // run forward pass for YOLOv5
    vector<Mat> outputs;
    yolo.forward(outputs, yolo.getUnconnectedOutLayersNames());

    // loop through the detections
    const int n_detections   = 25200;
    const int detection_size = 85;
    float* data              = (float*)outputs[0].data;

    for (size_t i = 0; i < n_detections; ++i) {

        // confidence
        float confidence = data[4];

        // continue if the confidence is above the threshold
        if (confidence >= CONFIDENCE_THRESHOLD) {

            // normalized coordinates of the detected bounding box
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            // actual coordinates of the detected bounding box
            int left   = int((x - 0.5 * w) * BLOB_X_FACTOR);
            int top    = int((y - 0.5 * h) * BLOB_Y_FACTOR);
            int width  = int(w * BLOB_X_FACTOR);
            int height = int(h * BLOB_Y_FACTOR);

            // store the detection in the vector "detections"
            Detection detection;
            detection.confidence   = confidence;
            detection.bounding_box = Rect(left, top, width, height);
            detections.push_back(detection);

        }

        // jump to the next detection
        data += detection_size;

    }
}

// function that draws the parking lots
void draw_parking_lots(Mat& image, vector<ParkingLot>& parking_lots, vector<Detection>& detections, float OVERLAP_THRESHOLD, int DETECTION_AREA_THRESHOLD)
{
    // loop through the parking lots
    for (size_t i = 0; i < parking_lots.size(); i++) {

        // parking lot
        Rect parking_lot = parking_lots[i].bounding_box;

        // boolean to know if the parking lot has been drawn or not
        bool drawn = false;

        // loop through the detections
        for (size_t j = 0; j < detections.size(); j++) {

            // detection
            Rect detection = detections[j].bounding_box;

            // continue if both of the following conditions are met:
            // 1) the area of the intersection of "parking_lot" and "detection" is greater than or equal to "OVERLAP_THRESHOLD" times the area of "parking_lot"
            // 2) the area of "detection" is less than "DETECTION_AREA_THRESHOLD" times the area of "parking_lot"
            if (((parking_lot & detection).area() >= (OVERLAP_THRESHOLD * parking_lot.area())) && (detection.area() < (DETECTION_AREA_THRESHOLD * parking_lot.area()))) {

                // draw the parking lot as occupied (RED)
                rectangle(image, parking_lot, RED, 1);
                drawn = true;
                break;

            }

        }

        if (drawn == false) {

            // draw the parking lot as free (GREEN)
            rectangle(image, parking_lot, GREEN, 1);

        }

        // slot ID
        string slot_id = parking_lots[i].slot_id;

        // draw the slot ID
        putText(image, slot_id, Point(parking_lot.x, parking_lot.y), FONT_ITALIC, 0.35, YELLOW, 1, LINE_AA);

    }
}

// main
int main(int argc, char** argv)
{
    // default input image path
    string INPUT_IMAGE_PATH = "CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/SUNNY/2015-11-12/camera4/2015-11-12_0916.jpg";
    
    // load the input image
    Mat input_image;
    if (argc > 1) {

        INPUT_IMAGE_PATH = argv[1];

    }
    input_image = imread(INPUT_IMAGE_PATH);

    // if the input image cannot be loaded: print indications of how to run this program
    if (input_image.empty()) {

        std::cerr << "\nInvalid input image!\n";
        std::cout << "Usage: " << argv[0] << " <path_to_input_image>\n";
        return -1;

    }

    // output image path
    string OUTPUT_IMAGE_PATH = INPUT_IMAGE_PATH.substr(0, INPUT_IMAGE_PATH.find(".jpg")) + "_output.jpg";

    // create a 4-dimensional BLOB from the input image
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(BLOB_WIDTH, BLOB_HEIGHT), Scalar(), true, false);

    // BLOB rescaling factors
    float BLOB_X_FACTOR = input_image.cols / BLOB_WIDTH;
    float BLOB_Y_FACTOR = input_image.rows / BLOB_HEIGHT;

    // load the YOLOv5 model
    Net yolo;
    yolo = readNet(YOLO_PATH);

    // camera number
    string CAMERA_NUMBER = INPUT_IMAGE_PATH.substr(INPUT_IMAGE_PATH.find("camera"), 7);

    // parking lots path
    string PARKING_LOTS_PATH = "CNR-EXT_FULL_IMAGE_1000x750/" + CAMERA_NUMBER + ".csv";

    // parking lots rescaling factors
    float PARKING_LOT_X_FACTOR = DOWNSAMPLED_IMAGE_WIDTH / ORIGINAL_IMAGE_WIDTH;
    float PARKING_LOT_Y_FACTOR = DOWNSAMPLED_IMAGE_HEIGHT / ORIGINAL_IMAGE_HEIGHT;

    // load the parking lots
    vector<ParkingLot> parking_lots;
    load_parking_lots(parking_lots, PARKING_LOTS_PATH, PARKING_LOT_X_FACTOR, PARKING_LOT_Y_FACTOR);

    // initial value of the thresholds
    int CONFIDENCE_THRESHOLD_PERCENTAGE = 1;
    int OVERLAP_THRESHOLD_PERCENTAGE    = 50;
    int DETECTION_AREA_THRESHOLD        = 5;

    // window name
    string WINDOW_NAME = "Parking Lot Occupancy Detection using YOLOv5 | " + INPUT_IMAGE_PATH;

    // create the window
    namedWindow(WINDOW_NAME, WINDOW_NORMAL);

    // create the trackbars
    createTrackbar(CONFIDENCE_THRESHOLD_TRACKBAR_NAME,     WINDOW_NAME, &CONFIDENCE_THRESHOLD_PERCENTAGE, TRACKBAR_MAX);
    createTrackbar(OVERLAP_THRESHOLD_TRACKBAR_NAME,        WINDOW_NAME, &OVERLAP_THRESHOLD_PERCENTAGE,    TRACKBAR_MAX);
    createTrackbar(DETECTION_AREA_THRESHOLD_TRACKBAR_NAME, WINDOW_NAME, &DETECTION_AREA_THRESHOLD,        TRACKBAR_MAX);

    // loop to display & refresh the output image until the user presses "q" or "Q"
    char key = 0;
    while (key != 'q' && key != 'Q') {

        // normalize the thresholds
        float CONFIDENCE_THRESHOLD = float(CONFIDENCE_THRESHOLD_PERCENTAGE) / TRACKBAR_MAX;
        float OVERLAP_THRESHOLD    = float(OVERLAP_THRESHOLD_PERCENTAGE) / TRACKBAR_MAX;

        // perform object detection using YOLOv5
        vector<Detection> detections;
        yolo_detect(blob, detections, yolo, CONFIDENCE_THRESHOLD, BLOB_X_FACTOR, BLOB_Y_FACTOR);

        // draw the parking lots
        Mat output_image = input_image.clone();
        draw_parking_lots(output_image, parking_lots, detections, OVERLAP_THRESHOLD, DETECTION_AREA_THRESHOLD);

        // show the output image
        imshow(WINDOW_NAME, output_image);

        // save the output image
        imwrite(OUTPUT_IMAGE_PATH, output_image);

        // get user key
        key = (char)waitKey(10);

    }

    return 0;
}