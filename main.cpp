//Some Windows Headers (For Time, IO, etc.)
#include <windows.h>
#include <mmsystem.h>

// OpenCV headers
#include <cv.h>
#include <highgui.h>

// OpenGL headers
#include <glew.h>
#include <freeglut.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "maths_funcs.h"	// Matrix math class
#include "teapot.h"			// teapot mesh

// Macro for indexing vertex buffer
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

using namespace cv;
using namespace std;

//// Open GL Data ////
GLuint shaderProgramID;

unsigned int teapot_vao = 0;

// Dimensions of OpenGL window (Same as webcam image dimensions)
int width = 640;
int height = 480;

// Camera clipping planes near and far
float n = 0.1f;
float f = 1000.0f;

float pointScale = 4.0f;

GLuint loc1;
GLuint loc2;

mat4 persp = zero_mat4();	// Prespective Matrix from intrinsic values
mat4 persp_proj;			// Perspective Projection Matrix
/////////////////////



//// OpenCV Data ////
// Chessboard pattern properties
int numCornersHor = 9;
int numCornersVer = 6;
int numSquares = numCornersHor * numCornersVer;;
Size boardSize = Size(numCornersHor, numCornersVer);

VideoCapture capture = VideoCapture(0);

vector<vector<Point3f>> object_points;				// Position of corners in virtual 3D space
vector<vector<Point2f>> image_points;				// Position of these corners on the camera image 
vector<Point2f> corners;

Mat image, gray_image;

double prevcam[] = {1016.72, 0.0,     336.26,
					0.0,     1015.85, 208.62,
					0.0,     0.0,     1.0};			// Default camera calibratiom (saved from previous results)

Mat intrinsic = Mat(3, 3, CV_64FC1, prevcam);		// Matrix of intrinsic properties of webcam
Mat distCoeffs;										// distortion coefficients
/////////////////////

// Shader Functions
#pragma region SHADER_FUNCTIONS

std::string readShaderSource(const std::string& fileName)
{
	std::ifstream file(fileName);
	if (file.fail())
		return false;

	std::stringstream stream;
	stream << file.rdbuf();
	file.close();

	return stream.str();
}


static void AddShader(GLuint ShaderProgram, const char* pShaderText, GLenum ShaderType)
{
	// create a shader object
	GLuint ShaderObj = glCreateShader(ShaderType);

	if (ShaderObj == 0) {
		fprintf(stderr, "Error creating shader type %d\n", ShaderType);
		exit(0);
	}
	std::string outShader = readShaderSource(pShaderText);
	const char* pShaderSource = outShader.c_str();

	// Bind the source code to the shader, this happens before compilation
	glShaderSource(ShaderObj, 1, (const GLchar**)&pShaderSource, NULL);
	// compile the shader and check for errors
	glCompileShader(ShaderObj);
	GLint success;
	// check for shader related errors
	glGetShaderiv(ShaderObj, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLchar InfoLog[1024];
		glGetShaderInfoLog(ShaderObj, 1024, NULL, InfoLog);
		fprintf(stderr, "Error compiling shader type %d: '%s'\n", ShaderType, InfoLog);
		exit(1);
	}
	// Attach the compiled shader object to the program object
	glAttachShader(ShaderProgram, ShaderObj);
}

GLuint CompileShaders()
{
	// Start the process of setting up shaders by creating a program ID
	// All the shaders are linked together into this ID
	shaderProgramID = glCreateProgram();
	if (shaderProgramID == 0) {
		fprintf(stderr, "Error creating shader program\n");
		exit(1);
	}

	// Create two shader objects, one for the vertex, and one for the fragment shader
	AddShader(shaderProgramID, "../Shaders/simpleVertexShader.txt", GL_VERTEX_SHADER);
	AddShader(shaderProgramID, "../Shaders/simpleFragmentShader.txt", GL_FRAGMENT_SHADER);

	GLint Success = 0;
	GLchar ErrorLog[1024] = { 0 };

	// After compiling all shader objects and attaching them to the program, we can finally link it
	glLinkProgram(shaderProgramID);
	// Check for program related errors
	glGetProgramiv(shaderProgramID, GL_LINK_STATUS, &Success);
	if (Success == 0) {
		glGetProgramInfoLog(shaderProgramID, sizeof(ErrorLog), NULL, ErrorLog);
		fprintf(stderr, "Error linking shader program: '%s'\n", ErrorLog);
		exit(1);
	}

	// Program has been successfully linked but needs to be validated to check whether the program can execute given the current pipeline state
	glValidateProgram(shaderProgramID);
	// Check for program related errors
	glGetProgramiv(shaderProgramID, GL_VALIDATE_STATUS, &Success);
	if (!Success) {
		glGetProgramInfoLog(shaderProgramID, sizeof(ErrorLog), NULL, ErrorLog);
		fprintf(stderr, "Invalid shader program: '%s'\n", ErrorLog);
		exit(1);
	}
	// Finally, use the linked shader program
	glUseProgram(shaderProgramID);

	return shaderProgramID;
}
#pragma endregion SHADER_FUNCTIONS

// VBO Functions
#pragma region VBO_FUNCTIONS

void generateObjectBufferTeapot() {
	GLuint vp_vbo = 0;

	loc1 = glGetAttribLocation(shaderProgramID, "vertex_position");
	loc2 = glGetAttribLocation(shaderProgramID, "vertex_normals");

	glGenBuffers(1, &vp_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vp_vbo);
	glBufferData(GL_ARRAY_BUFFER, 3 * teapot_vertex_count * sizeof (float), teapot_vertex_points, GL_STATIC_DRAW);
	GLuint vn_vbo = 0;
	glGenBuffers(1, &vn_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vn_vbo);
	glBufferData(GL_ARRAY_BUFFER, 3 * teapot_vertex_count * sizeof (float), teapot_normals, GL_STATIC_DRAW);

	glGenVertexArrays(1, &teapot_vao);
	glBindVertexArray(teapot_vao);

	glEnableVertexAttribArray(loc1);
	glBindBuffer(GL_ARRAY_BUFFER, vp_vbo);
	glVertexAttribPointer(loc1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(loc2);
	glBindBuffer(GL_ARRAY_BUFFER, vn_vbo);
	glVertexAttribPointer(loc2, 3, GL_FLOAT, GL_FALSE, 0, NULL);
}


#pragma endregion VBO_FUNCTIONS

void display()
{
	// Transformation matrix for object being rendered
	mat4 local1 = zero_mat4();

	// Get image from webcam
	capture >> image;

	bool found = findChessboardCorners(image, boardSize, corners, CALIB_CB_FAST_CHECK);
	drawChessboardCorners(image, boardSize, corners, found);

	if(found)
	{
		vector<Point3f> obj;
		for(int j=0;j<numSquares;j++)
			obj.push_back(Point3f(j/numCornersHor, j%numCornersHor, 0.0f)*pointScale);

		Mat rvec;	// Rotation vector
		Mat tvec;	// Translation vector

		// Get extrinsic properties rvec and tvec
		solvePnP(obj, corners, intrinsic, distCoeffs, rvec, tvec, false, CV_EPNP);

		// Convert rvec to Rotation matrix using rodrigues
		Mat rotation(3, 3, CV_64FC1);
		Rodrigues(rvec, rotation);
		
		// Combine matrices
		// Column0
		local1.m[0] = rotation.at<double>(0,0);
		local1.m[1] = rotation.at<double>(1,0);
		local1.m[2] = rotation.at<double>(2,0);
		local1.m[3] = 0.0f;

		// Column1
		local1.m[4] = rotation.at<double>(0,1);
		local1.m[5] = rotation.at<double>(1,1);
		local1.m[6] = rotation.at<double>(2,1);
		local1.m[7] = 0.0f;

		// Column2
		local1.m[8] = rotation.at<double>(0,2);
		local1.m[9] = rotation.at<double>(1,2);
		local1.m[10] = rotation.at<double>(2,2);
		local1.m[11] = 0.0f;

		// Column3
		local1.m[12] = tvec.at<double>(0);
		local1.m[13] = tvec.at<double>(1);
		local1.m[14] = -tvec.at<double>(2); // Need to flip z-axis translation. This matches flipping that we do on the webcam image next
		local1.m[15] = 1.0f;
	}

	// OpenGL starts pixels from top-left, opencv start from bottom left
	// Fix this by flipping the webcam image
	cvtColor(image,image,CV_RGB2BGR);
	flip(image,image,0);

	// Clear frame
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Draw webcam image
	glDrawPixels(width,height,GL_RGB,GL_UNSIGNED_BYTE, image.data);

	// Clear depth again because glDrawPixels affects depth buffer
	glClear (GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glUseProgram(shaderProgramID);
	
	//Declare uniform variables that will be used in shader
	int matrix_location = glGetUniformLocation(shaderProgramID, "model");
	int view_mat_location = glGetUniformLocation(shaderProgramID, "view");
	int proj_mat_location = glGetUniformLocation(shaderProgramID, "proj");

	if(found)
	{
		// Draw Teapot
		mat4 view = identity_mat4();
		glUniformMatrix4fv (proj_mat_location, 1, GL_FALSE, persp_proj.m);
		glUniformMatrix4fv (view_mat_location, 1, GL_FALSE, view.m);
		glUniformMatrix4fv (matrix_location, 1, GL_FALSE, local1.m);
		glDrawArrays (GL_TRIANGLES, 0, teapot_vertex_count);
	}

	glutSwapBuffers();
}

void updateScene()
{
	// Draw the next frame
	glutPostRedisplay();
}

void init()
{
	cout << "Calibrate? " << endl << "1) Yes" << endl << "2) No" << endl;
	char key;
	cin >> key;

	// Intrinsic camera matrix setup
	intrinsic.ptr<float>(0)[0] = 1;
	intrinsic.ptr<float>(1)[1] = 1;

	// Calibrate for new cameras
	// I have my own camera's intrinsic values calculated and saved already
	if(key == '1')
	{
		cout << "How many samples?" << endl;
		int samples;
		cin >> samples;

		cout << "Press spacebar to sample" << endl;

		int sampleCount = 0;

		capture >> image;

		vector<Point3f> obj;
		for(int j=0;j<numSquares;j++)
			obj.push_back(Point3f(j/numCornersHor, j%numCornersHor, 0.0f)*pointScale);

		while (sampleCount < samples)
		{
			cvtColor(image, gray_image, CV_BGR2GRAY);

			bool found = findChessboardCorners(image, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
 
			if(found)
			{
				cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
				drawChessboardCorners(image, boardSize, corners, found);
			}

			imshow("Calibrate Camera", image);
 
			capture >> image;
 
			int key = waitKey(1);
 
			if(key==' ' && found!=0)
			{
				image_points.push_back(corners);
				object_points.push_back(obj);
				sampleCount++;

				cout << "Sample stored" << endl;
			}
		}

		cout << "Finished sampling" << endl;

		vector<Mat> rvecs;	// rotation vectors
		vector<Mat> tvecs;	// translation vectors

		calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);

		cout << intrinsic <<endl;
	}

	destroyAllWindows();

	// Move intrinsic values to a 4x4 perspective matrix
	persp.m[0] = intrinsic.at<double>(0,0);
	persp.m[5] = intrinsic.at<double>(1,1);
	persp.m[8] = -intrinsic.at<double>(0,2);
	persp.m[9] = -intrinsic.at<double>(1,2);
	persp.m[10] = n+f;
	persp.m[11] = -1.0f;
	persp.m[14] = n*f;

	// Compute normalized device coords
	mat4 NDC = orthographic(0, (float)width, (float)height, 0.0f, n, f);

	// Use these two matrices to compute the projection matrix
	persp_proj = NDC*persp;

	// Set up the shaders
	GLuint shaderProgramID = CompileShaders();
	// load teapot mesh into a vertex buffer array
	generateObjectBufferTeapot();
}

void keypress(unsigned char key, int x, int y)
{
	// Adjust scale of virtual 3D points
	if(key == 'w'){
		pointScale += 1.0f;
		cout << "pointScale= " << pointScale << endl;
	}

	if(key=='s'){
		pointScale -= 0.1f;
		cout << "pointScale= " << pointScale << endl;
	}
}

int main(int argc, char** argv)
{
	// Set up the window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(width, height);
	glutCreateWindow("AR");

	// Tell glut where the display function is
	glutDisplayFunc(display);
	glutIdleFunc(updateScene);
	glutKeyboardFunc(keypress);

	// A call to glewInit() must be done after glut is initialized!
	GLenum res = glewInit();
	// Check for any errors
	if (res != GLEW_OK) {
		fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
		return 1;
	}

	init();
	
	glutMainLoop();
 
    return 0;
}

