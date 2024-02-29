
#include <afx.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <direct.h>
#include <io.h>
#include <atlstr.h>


#include <cassert>
#include <itkImage.h>
#include <itkImageSeriesReader.h>
#include <itkGDCMImageIO.h>
#include <itkImageFileReader.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkIdentityTransform.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkImageFileWriter.h>
#include <itkNIFTIImageIO.h>
#include <itkNIFTIImageIOFactory.h>
#include <itkImageToVTKImageFilter.h>
#include <itkGDCMImageIO.h>
#include <vtkDataSetWriter.h>
#include <itkVTKImageToImageFilter.h>
#include <itkNumericSeriesFileNames.h>
#include <vector>
#include <array>
#include <thread>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkNIFTIImageReader.h>


#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>


#include <vtkImageReslice.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesWriter.h>
#include <vtkNIFTIImageWriter.h>
#include <vtkInformation.h>
#include <vtkMath.h>
#include <vtkTransform.h>

#include <opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/features2d/features2d.hpp>



#include <vector>
#include <numeric>
#include <stdio.h>
#include <algorithm>


#include <Eigen>



#define nPoint 5 
#define imgX 320
#define imgY 320
#define imgZ 120

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;
using namespace std;

using ImageType2U = itk::Image<unsigned short, 2>;
using ImageType3F = itk::Image<float, 3>;
using ImageType2F = itk::Image<float, 2>;
using ReaderType = itk::ImageFileReader<ImageType3F>;


int orgImgX = 320;
int orgImgY = 320; int orgImgZ =120;


struct mprOutStruct {
	array<float, 3> point1;
	array<float, 3> point2;
	array<float, 3> point3;
	array<float, 3> point4;
	array<float, 3> point5;
	vtkSmartPointer<vtkImageData> resliceAxial;
	vtkSmartPointer<vtkImageData> resliceSagittal;
	vtkSmartPointer<vtkImageData> resliceCoronal;
	double tran_angle[3];
	double tran_origin[3];
};

class Logger : public ILogger
{
	virtual void log(Severity severity, const char* msg) noexcept override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			cout << msg << endl;
	}
} gLogger;

void SubtractVectors(const array<float, 3>& a, const array<float, 3>& b, array<float, 3>& result)
{
	result[0] = a[0] - b[0];
	result[1] = a[1] - b[1];
	result[2] = a[2] - b[2];
}

// 向量归一化
void NormalizeVector(array<float, 3>& vec)
{
	float norm = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
	vec[0] /= norm;
	vec[1] /= norm;
	vec[2] /= norm;
}

// 计算向量叉乘
array<float, 3> cal_norm_vec(array<float, 3> v1, array<float, 3> v2) {
	array<float, 3> norm_vec;
	norm_vec[0] = v1[1] * v2[2] - v1[2] * v2[1];
	norm_vec[1] = v1[2] * v2[0] - v1[0] * v2[2];
	norm_vec[2] = v1[0] * v2[1] - v1[1] * v2[0];
	NormalizeVector(norm_vec);
	return norm_vec;
}

// 向量取反
array<float, 3> apply_neg(array<float, 3> vec) {
	array<float, 3> neg_vec;
	neg_vec[0] = -vec[0];
	neg_vec[1] = -vec[1];
	neg_vec[2] = -vec[2];
	return neg_vec;
}

array<float, 3> ImageToPhysicalPoint(const array<int, 3>& imagePoint, const array<float, 3>& origin, const array<float, 3>& spacing)
{
	array<float, 3> physicalPoint;
	for (int i = 0; i < 3; ++i) {
		physicalPoint[i] = origin[i] + imagePoint[i] * spacing[i];
	}
	return physicalPoint;
}


// 将vtk世界坐标系还原回vtk坐标系
array<float, 3> PhysicalPointToImage(const array<float, 3>& physicalPoint, const array<float, 3>& origin, const array<float, 3>& spacing)
{
	array<float, 3> imagePoint;
	for (int i = 0; i < 3; ++i) {
		imagePoint[i] = (physicalPoint[i] - origin[i]) / spacing[i];
	}
	return imagePoint;
}




/*
Function: ImageToPatientCoord

Descirption: transfer vtk coord to patient coord(dcm)
@param imagePoint: vtk coord, save in x-y-z(vtk coord);
@param origin: origin pt coord at 3d patient data;
@param spacing: just spacing, save in x-y-z(patient coord).
@param sliceflag: indicate the scan direction, L/R or R/L.

return patient coord.
*/
array<float, 3> ImageToPatientCoord(const array<float, 3>& imagePoint, const array<float, 3>& origin, const array<float, 3>& spacing, int sliceflag=1)
{
	array<float, 3> PatientPoint;

	if(sliceflag) // normal situation, scan direction L -> R
		PatientPoint[0] = origin[0] - imagePoint[2] * spacing[2];
	else // rare situation, scan direction R -> L
		PatientPoint[0] = origin[0] + imagePoint[2] * spacing[2];//(origin[0] + float(sliceDirection)) - imagePoint[2] * spacing[2];
	PatientPoint[1] = origin[1] + imagePoint[0] * spacing[0];
	PatientPoint[2] = origin[2] - imagePoint[1] * spacing[1];

	return PatientPoint;
}



/************************  ikt img 格式转换 F->U *********************************/
ImageType2U::Pointer formatchange(ImageType2F::Pointer itkImageF, ImageType2U::SizeType size) {
	ImageType2U::Pointer itkImage = ImageType2U::New();
	// 设置ITK图像的大小和像素数据
	itkImage->SetRegions(size);
	itkImage->Allocate();

	for (int y = 0; y < size[1]; ++y)
		for (int x = 0; x < size[0]; ++x)
			itkImage->SetPixel({ x, y }, static_cast<unsigned short>(itkImageF->GetPixel({ x,y })));

	itkImage->SetSpacing(itkImageF->GetSpacing());
	itkImageF->SetOrigin(itkImageF->GetOrigin());
	return itkImage;
}



// 线程函数，计算每个通道的最大值索引
void calculateMaxIndex(const itk::Image<float, 3>::Pointer& channelImage, array<int, 3>& maxIndex)
{
	using ImageIndexType = ImageType3F::IndexType;
	float maxValue = numeric_limits<float>::min();

	// 遍历图像像素，找到最大值索引
	for (int z = 0; z < orgImgZ; ++z) // imgZ
	{
		for (int y = 0; y <  orgImgY; ++y) //imgY
		{
			for (int x = 0; x < orgImgX; ++x) // imgX
			{
				ImageIndexType currentIndex;
				currentIndex[0] = x;
				currentIndex[1] = y;
				currentIndex[2] = z;

				float pixelValue = channelImage->GetPixel(currentIndex);
				if (pixelValue > maxValue)
				{
					maxValue = pixelValue;
					maxIndex = { x, y, z };
				}
			}
		}
	}
}

ImageType3F::Pointer resizeImg(ImageType3F::Pointer oriImg, ImageType3F::SizeType targetSize)
{
	
	using ResampleFilterType = itk::ResampleImageFilter<ImageType3F, ImageType3F>;
	// typedef itk::ResampleImageFilter<ImageType3F, ImageType3F> ResampleFilterType;
	ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
	resampleFilter->SetInput(oriImg);
	resampleFilter->SetSize(targetSize);
	// 计算resize后的spacing
	ImageType3F::SizeType origintSize = oriImg->GetLargestPossibleRegion().GetSize();
	ImageType3F::SpacingType targetSpacing;
	for (int i = 0; i < 3; ++i) {
		targetSpacing[i] = oriImg->GetSpacing()[i] * origintSize[i] / targetSize[i];
	}
	resampleFilter->SetOutputSpacing(targetSpacing);

	// 设置输出原点与输入一致
	resampleFilter->SetOutputOrigin(oriImg->GetOrigin());
	resampleFilter->SetOutputDirection(oriImg->GetDirection());

	// 使用线性插值
	typedef itk::LinearInterpolateImageFunction<ImageType3F, double> InterpolatorType;
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	resampleFilter->SetInterpolator(interpolator);

	// 更新输出图像
	resampleFilter->Update();

	return resampleFilter->GetOutput();
}

ImageType2F::Pointer resizeImg(ImageType2F::Pointer oriImg, ImageType2F::SizeType targetSize)
{

	typedef itk::ResampleImageFilter<ImageType2F, ImageType2F> ResampleFilterType;
	ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
	resampleFilter->SetInput(oriImg);
	resampleFilter->SetSize(targetSize);
	// 计算resize后的spacing
	ImageType2U::SizeType origintSize = oriImg->GetLargestPossibleRegion().GetSize();
	ImageType2F::SpacingType targetSpacing;
	for (int i = 0; i < 2; ++i) {
		targetSpacing[i] = oriImg->GetSpacing()[i] * origintSize[i] / targetSize[i];
	}
	resampleFilter->SetOutputSpacing(targetSpacing);

	// 设置输出原点与输入一致
	resampleFilter->SetOutputOrigin(oriImg->GetOrigin());
	resampleFilter->SetOutputDirection(oriImg->GetDirection());

	// 使用线性插值
	typedef itk::LinearInterpolateImageFunction<ImageType2F, double> InterpolatorType;
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	resampleFilter->SetInterpolator(interpolator);

	// 更新输出图像
	resampleFilter->Update();

	return resampleFilter->GetOutput();
}



/************************后处理获得关键点坐标*********************************/
vector<array<int, 3>> processOutputBuffer(const float* output_buffer)
{
	using ResampleFilterType = itk::ResampleImageFilter<ImageType3F, ImageType3F>;
	using InterpolatorType = itk::LinearInterpolateImageFunction<ImageType3F, double>;

	// 创建线程列表和最大值索引结果数组
	vector<thread> threads;
	vector<array<int, 3>> maxIndices(nPoint);

	for (int channel = 0; channel < nPoint; ++channel)
	{
		// 创建每个通道的图像
		ImageType3F::Pointer channelImage = ImageType3F::New();
		ImageType3F::RegionType region;
		ImageType3F::IndexType start;
		start.Fill(0);
		ImageType3F::SizeType size;
		size[0] = imgX;
		size[1] = imgY;
		size[2] = imgZ;
		region.SetIndex(start);
		region.SetSize(size);
		channelImage->SetRegions(region);
		channelImage->Allocate();

		// 复制输出缓冲区的数据到通道图像
		float* imageBuffer = channelImage->GetBufferPointer();
		const float* channelBuffer = output_buffer + channel * imgX * imgY * imgZ;
		memcpy(imageBuffer, channelBuffer, imgX * imgY * imgZ * sizeof(float));

		//// 调整图像大小
		ImageType3F::SizeType targetSize;
		targetSize[0] = orgImgX;
		targetSize[1] = orgImgY;
		targetSize[2] = orgImgZ;
		ImageType3F::Pointer resampledImg = resizeImg(channelImage, targetSize);

		// 启动线程，求最大值索引
		threads.emplace_back(calculateMaxIndex, resampledImg, ref(maxIndices[channel]));
		// threads.emplace_back(calculateMaxIndex, channelImage, ref(maxIndices[channel]));
	}

	// 等待所有线程完成
	for (auto& thread : threads)
	{
		thread.join();
	}
	return maxIndices;
}




/************************ 保存dcm *********************************/
/*
函数说明：用于修改必要dicom tag并保存对应的vtk切片至dicom格式

itk::MetaDataDictionary& metaData_saved：
int InstanceNumber: dicom 图像顺序tag，根据协商的保存顺序，九张切片依次为3轴3冠3矢
*/

void SaveAsDCM(const vtkSmartPointer<vtkImageData>& imageData, itk::MetaDataDictionary& metaData_saved, 
	itk::GDCMImageIO::Pointer dcmIO, const string& outputPath, int InstanceNumber = 1)
{
	// 转换VTK图像为ITK图像
	ImageType2F::Pointer itkImageF = ImageType2F::New();

	// 设置ITK图像的大小和像素数据
	ImageType2U::SizeType size;
	size[0] = imageData->GetDimensions()[0];
	size[1] = imageData->GetDimensions()[1];

	itkImageF->SetRegions(size);
	itkImageF->Allocate();

	if (InstanceNumber < 0) {
		for (int y = 0; y < size[1]; ++y)
			for (int x = 0; x < size[0]; ++x)
				itkImageF->SetPixel({ int(size[0])-x-1, y }, static_cast<float>(imageData->GetScalarComponentAsFloat(x, y, 0, 0)));
	}
	else {
		for (int y = 0; y < size[1]; ++y)
			for (int x = 0; x < size[0]; ++x)
				itkImageF->SetPixel({ x, y }, static_cast<float>(imageData->GetScalarComponentAsFloat(x, y, 0, 0)));
		//itkImageF->SetPixel({ x,int(size[1]) - y - 1},
	}
	

	// 设置原点和间距
	ImageType2F::PointType origin;
	origin[0] = imageData->GetOrigin()[0];
	origin[1] = imageData->GetOrigin()[1];
	itkImageF->SetOrigin(origin);
	ImageType2F::SpacingType spacing;
	spacing[0] = imageData->GetSpacing()[0];;
	spacing[1] = imageData->GetSpacing()[1];
	itkImageF->SetSpacing(spacing);

	// 根据pixelspacing重新变化图像矩阵大小，转变为近似pixelspacing图像
	if (spacing[0] != spacing[1]) {
		if (spacing[0] > spacing[1])
			size[0] = int(round(double(size[0])*double(spacing[0]) / double(spacing[1])));
		else
			size[1] = int(round(double(size[1])*double(spacing[1]) / double(spacing[0])));

		itkImageF = resizeImg(itkImageF, size);
	}

	// 图像类型转换，转为unsigned short 输出图像
	ImageType2U::Pointer itkImage = formatchange(itkImageF, size);

	// 创建Dcm图像IO对象
	/*using ImageIOType_dcm = itk::GDCMImageIO;
	ImageIOType_dcm::Pointer dcmIO = ImageIOType_dcm::New();*/

	// 创建Dcm图像写入器
	using WriterType = itk::ImageFileWriter<ImageType2U>;
	WriterType::Pointer writer_dcm = WriterType::New();

	// 设置输出路径
	writer_dcm->SetFileName(outputPath);

	// 用于拼接字符串临时变量
	std::ostringstream value;
	value.str(""); // 清空	

	if (InstanceNumber == 2) {
		// some tag just need to update once during all slices

		// 开始逐一更新必要的DICOM标签
		itk::EncapsulateMetaData<string>(metaData_saved, "0008|0008", "Localizer"); // May not need to change
		// string seriesDescription = "Localizer";
		// itk::EncapsulateMetaData<string>(metaData_saved, "0008|103e", seriesDescription); // May not need to change
		
		// set for ANKE Apex localizer recognization
		itk::EncapsulateMetaData<string>(metaData_saved, "0018|0024", "grscout");
		// 理论上以后无需继续设置厚度 //string sliceThickness = "2.0";	
		itk::EncapsulateMetaData<string>(metaData_saved, "0018|0050", "2.0");
		// series total Number
		string NumberofSlices = "9";
		itk::EncapsulateMetaData<string>(metaData_saved, "0054|0081", NumberofSlices);


		// Test
		value.str("");
		for (int i = 0; i < 5; i++)
			value << "0" << "\\";
		value << "0" << "\\";
		itk::EncapsulateMetaData<string>(metaData_saved, "0065|0022", value.str());
		value.str("");

		// APEX Exam judge for localizer tag
		itk::EncapsulateMetaData<string>(metaData_saved, "0065|0015", value.str());
		itk::EncapsulateMetaData<string>(metaData_saved, "0065|0016", value.str());
		itk::EncapsulateMetaData<string>(metaData_saved, "0065|0017", value.str());

		// Set a better default WL/WW empirically
		value << "3700.0\\4094.0";
		itk::EncapsulateMetaData<string>(metaData_saved, "0028|1051", value.str());
		value.str("");
		value << "1800.0\\2048.0";
		itk::EncapsulateMetaData<string>(metaData_saved, "0028|1050", value.str());
		value.str("");

	}


	// SOPInstanceUId
	std::string SOPInstanceUId; std::string key = "0008|0018";
	itk::ExposeMetaData(metaData_saved, key, SOPInstanceUId);
	SOPInstanceUId.erase(SOPInstanceUId.end() - 2, SOPInstanceUId.end());
	value << SOPInstanceUId << "0" << InstanceNumber;
	itk::EncapsulateMetaData<string>(metaData_saved, "0008|0018", value.str());
	// 更新fov
	value.str(""); value << round(double(size[0])*itkImage->GetSpacing()[0])
		<< "\\" << round(size[1] * itkImage->GetSpacing()[1]);
	itk::EncapsulateMetaData<string>(metaData_saved, "0018|7030", value.str());
	itk::EncapsulateMetaData<string>(metaData_saved, "0065|0018", value.str());
	// 切片在病人坐标系下方向
	string PatientOrientation;
	if (InstanceNumber < 4) {
		value.str(""); value << "TRAN";
		PatientOrientation = "A\\R";// 根据轴冠矢状确定
	}
	else if (InstanceNumber < 7) {
		value.str(""); value << "SAGI";
		PatientOrientation = "H\\A";
	}
	else {
		value.str(""); value << "CORO";
		PatientOrientation = "H\\R";
	}
	itk::EncapsulateMetaData<string>(metaData_saved, "0020|0020", PatientOrientation);
	itk::EncapsulateMetaData<string>(metaData_saved, "0065|0014", value.str());
	// InstanceNumber, 要按顺序排布，三张轴三张冠三张矢
	value.str(""); value << InstanceNumber;
	itk::EncapsulateMetaData<string>(metaData_saved, "0020|0013", value.str());
	// 更新切片dicom的矩阵长宽
	unsigned short Rows = unsigned short(size[0]);
	string Cols = to_string(double(size[1]));
	itk::EncapsulateMetaData<unsigned short>(metaData_saved, "0028|0010", Rows);
	itk::EncapsulateMetaData<string>(metaData_saved, "0028|0011", Cols);
	// PixelSpacing
	value.str(""); value << std::setiosflags(std::ios::fixed) << std::setprecision(5) <<
		itkImage->GetSpacing()[0] << "\\" << itkImage->GetSpacing()[1];
	itk::EncapsulateMetaData<string>(metaData_saved, "0028|0030", value.str());
	//PixelAspectRatio
	value.str(""); value << round(100 * itkImage->GetSpacing()[0]) <<
		"\\" << round(100 * itkImage->GetSpacing()[1]);
	itk::EncapsulateMetaData<string>(metaData_saved, "0028|0034", value.str());



	// 图像长宽；double2string？
	value.str(""); value << size[0] << "\\" << size[1];
	itk::EncapsulateMetaData<string>(metaData_saved, "0065|0010", value.str());
	value.clear();


	// 执行写入操作
	// 可选择覆盖itkImage的meta利用其保存，或者直接用IO的metadata进行保存
	// itkImage->SetMetaDataDictionary(metaData_saved);writer_dcm->SetUseInputMetaDataDictionary(1);
	dcmIO->SetMetaDataDictionary(metaData_saved);
	writer_dcm->SetImageIO(dcmIO);
	writer_dcm->SetInput(itkImage);
	writer_dcm->SetUseInputMetaDataDictionary(0);
	writer_dcm->Update();
}



/***********************dcm图像加载*********************************/
ImageType3F::Pointer dcmDataLoad(const string& dcmPath, itk::MetaDataDictionary& metaori)
{
	size_t lastSlash = dcmPath.find_last_of("/\\");
	string pid = dcmPath.substr(lastSlash + 1);

	using ReaderType = itk::ImageSeriesReader<ImageType3F>;
	using NamesGeneratorType = itk::NumericSeriesFileNames;

	ReaderType::Pointer reader = ReaderType::New();
	NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();

	namesGenerator->SetStartIndex(0);
	namesGenerator->SetEndIndex(118);
	namesGenerator->SetIncrementIndex(1);
	namesGenerator->SetSeriesFormat(dcmPath + "/" + pid + "%04d.dcm");

	const std::vector<std::string>& fileNames = namesGenerator->GetFileNames();
	reader->SetFileNames(fileNames);

	itk::GDCMImageIO::Pointer dicomIO = itk::GDCMImageIO::New();
	// 获取privatetag
	dicomIO->LoadPrivateTagsOn();
	reader->SetImageIO(dicomIO);
	reader->Update();
	metaori = dicomIO->GetMetaDataDictionary();
	return reader->GetOutput();
}


ImageType3F::Pointer dcmDataLoad_new(const string& dcmPath, itk::MetaDataDictionary& metaori) {
	// 获取DICOM系列的文件名
	itk::GDCMSeriesFileNames::Pointer namesGenerator = itk::GDCMSeriesFileNames::New();
	namesGenerator->SetUseSeriesDetails(true);
	namesGenerator->AddSeriesRestriction("0008|0021");

	namesGenerator->SetDirectory(dcmPath);

	typedef std::vector<std::string> SeriesIdContainer;
	const SeriesIdContainer & seriesUID = namesGenerator->GetSeriesUIDs();

	// 选择第一个DICOM系列
	std::string seriesIdentifier = seriesUID.begin()->c_str();

	// 获取该系列的所有文件名
	std::vector<std::string> fileNames = namesGenerator->GetFileNames(seriesIdentifier);

	// 创建DICOM图像读取器
	itk::ImageSeriesReader<ImageType3F>::Pointer reader = itk::ImageSeriesReader<ImageType3F>::New();

	itk::GDCMImageIO::Pointer dicomIO = itk::GDCMImageIO::New();
	// 获取privatetag
	dicomIO->LoadPrivateTagsOn();

	reader->SetFileNames(fileNames);
	reader->SetImageIO(dicomIO);
	reader->Update();
	metaori = dicomIO->GetMetaDataDictionary();
	return reader->GetOutput();
}




/*
Function: dcmDataLoad

Descirption: Load 3d img data with itk.
@param dcmPath: temp folder contain ori tfe 3d dcms;
@param metaori: Contains metadata for later save in dicom;
@param savePath: folder to save the 9 mpr dcm data.
@param fileNum: total dcm img num.

return 3D itk img data.
*/
ImageType3F::Pointer dcmDataLoad(const string& dcmPath, itk::MetaDataDictionary& metaori, string& savePath, int fileNum = 120)
{
	//size_t lastSlash = dcmPath.find_last_of("/\\");
	//string pid = dcmPath.substr(lastSlash + 1);
	size_t lastSlash = savePath.find_last_of("/\\");
	string pid = savePath.substr(lastSlash + 1);

	//std::ostringstream value;
	//value.str(""); // clear, or inital
	//for (auto s1 = dcmPath.begin(); s1 != dcmPath.end(); ++s1)
	//{
	//	if (*s1 == '\\') {
	//		value << "/";			
	//	}
	//	else {
	//		value << *s1;
	//	}
	//}


	std::ostringstream value1;
	value1.str(""); // clear, or inital
	int splashflag = 0;
	for (int id = 0; id < dcmPath.length(); id++) {
		//int flag = strcmp(&dcmPath[id], '\\');
		if (dcmPath[id] != '\\'){//flag!=1) {
			splashflag = 0;
			value1 << dcmPath[id];
		}
		else {
			if (splashflag == 0) {
				value1 << "/";
				splashflag = 1;
			}
		}
	}


	using ReaderType = itk::ImageSeriesReader<ImageType3F>;
	using NamesGeneratorType = itk::NumericSeriesFileNames;

	ReaderType::Pointer reader = ReaderType::New();
	NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();

	namesGenerator->SetStartIndex(0);
	namesGenerator->SetEndIndex(fileNum-1);
	namesGenerator->SetIncrementIndex(1);
	string test = value1.str() + "/" + pid + "%04d.dcm";
	namesGenerator->SetSeriesFormat(test);

	const std::vector<std::string>& fileNames = namesGenerator->GetFileNames();
	reader->SetFileNames(fileNames);

	itk::GDCMImageIO::Pointer dicomIO = itk::GDCMImageIO::New();
	// 获取privatetag
	dicomIO->LoadPrivateTagsOn();
	reader->SetImageIO(dicomIO);
	reader->Update();
	metaori = dicomIO->GetMetaDataDictionary();
	return reader->GetOutput();
}





/*
Function: noiseDetect

Descirption: Load 3d img data & decrease the noise effect created by tfe sequence in 2 blank side.
@param Image: 3d itk img builded to mpr later;
@param noiseCof: noise decreasing coef.

return: 3D itk img data.
*/
ImageType3F::Pointer noiseDetect(ImageType3F::Pointer Image, float noiseCof = .1) {
	ImageType3F::SizeType origintSize = Image->GetLargestPossibleRegion().GetSize();
	orgImgX = origintSize[2];
	orgImgY = origintSize[1];
	orgImgZ = origintSize[0];

	int rightBorder = 0, leftBorder = 0;

	

	int grouplen = int(round(0.4*orgImgX));
	Mat areacountR = Mat::zeros(1, grouplen, CV_32SC1);
	Mat areacountL = Mat::zeros(1, grouplen, CV_32SC1);



	//Mat tempslice = Mat::zeros(Size(orgImgY, orgImgZ), CV_32FC1);
	//Mat tempsliceBW = Mat::zeros(Size(orgImgY, orgImgZ), CV_8UC1);

	//for (int x = 0; x < orgImgZ; x++) {
	//	for (int y = 0; y < orgImgY; y++) {
	//
	//		ImageType3F::IndexType pixelIndex = { {x,y,orgImgX/2 -1 } };
	//		float a = Image->GetPixel(pixelIndex);
	//		tempslice.at<float>(y, x) = a;// Image->GetPixel(pixelIndex);
	//		if (a > 600) {
	//			tempsliceBW.at<uchar>(y, x) = 1;
	//		}
	//		
	//	}
	//}
	//Mat tempsliceBW1;
	double bwthres = 600;
	//bwthres = threshold(tempslice, tempsliceBW1, 0,4095,THRESH_OTSU | THRESH_BINARY);
	

	for (int sliceNum = 0; sliceNum< grouplen; sliceNum++) {

		// Mat tempslice = Mat::zeros(Size(orgImgY, orgImgZ), CV_32FC1);
		// Mat tempsliceBW = Mat::zeros(Size(orgImgY, orgImgZ), CV_8UC1);

		for (int x = 0; x < orgImgZ; x++) {
			for (int y = 0; y < orgImgY; y++) {
				// right side account
				
				ImageType3F::IndexType pixelIndex = { {x,y,sliceNum} };
				float a = Image->GetPixel(pixelIndex);
				// tempslice.at<float>(y, x) = a;// Image->GetPixel(pixelIndex);
				if (a > bwthres) {
					areacountR.at<int>(0, sliceNum) += 1;
					// tempsliceBW.at<uchar>(y, x) = 1;
				}

				// Left side account
				pixelIndex = { {x,y,orgImgX - grouplen + sliceNum - 3} };
				a = Image->GetPixel(pixelIndex);
				// tempslice.at<float>(y, x) = a;
				if (a > bwthres) {
					areacountL.at<int>(0, sliceNum) += 1;
					// tempsliceBW.at<uchar>(y, x) = 1;
				}
			}
		}
		// int stopflagview = 0;
	}

	Point minLoc;
	cv::minMaxLoc(areacountR, NULL, NULL,&minLoc,NULL);
	rightBorder = int(minLoc.x);
	cv::minMaxLoc(areacountL, NULL, NULL, &minLoc, NULL);
	leftBorder = orgImgX - (orgImgX + (int(minLoc.x) - 3) - grouplen);//orgImgX+(int(minLoc.x)-3) - grouplen;
	
	//if (rightBorder > leftBorder) {
		//leftBorder = rightBorder;
	//}
	//else {
		//rightBorder = leftBorder;
	//}

	// reset the noise part in itk 3d imgdata before
	// we gonna to resclice it to 2d localizer.
	float coef = 0;
	//coef = 1.05 * 1 / float(leftBorder + 5);


	//// left border denoise
	for (int sliceNum = 0; sliceNum < leftBorder + 2; sliceNum++) {
		for (int x = 0; x < orgImgZ; x++) {
			for (int y = 0; y < orgImgY; y++) {
				coef = .65 * (sliceNum+1) / float(leftBorder + 6);
				ImageType3F::IndexType pixelIndex = { {x,y,orgImgX - sliceNum-1} };
				Image->SetPixel(pixelIndex, coef*Image->GetPixel(pixelIndex));
			}
		}
		
	}

	coef = (1.05 * float((rightBorder + 4)/2) / float(rightBorder + 5));
	// right border denoise
	for (int sliceNum = 0; sliceNum < rightBorder + 2; sliceNum++) {
		for (int x = 0; x < orgImgZ; x++) {
			for (int y = 0; y < orgImgY; y++) {
				coef = .65 * (sliceNum+1) / float(rightBorder + 6);
				ImageType3F::IndexType pixelIndex = { {x,y,sliceNum} };
				Image->SetPixel(pixelIndex, coef*Image->GetPixel(pixelIndex));
			}
		}
	}


	return Image;
}




/************************图像预处理*********************************/
ImageType3F::Pointer dataPreprocess(ImageType3F::Pointer Image)
{
	// 图像归一化
	typedef itk::RescaleIntensityImageFilter<ImageType3F, ImageType3F> RescalerType;
	RescalerType::Pointer rescaler = RescalerType::New();
	rescaler->SetInput(Image);
	rescaler->SetOutputMinimum(0.0);
	rescaler->SetOutputMaximum(1.0);
	rescaler->Update();

	ImageType3F::SizeType origintSize = rescaler->GetOutput()->GetLargestPossibleRegion().GetSize();
	orgImgX = origintSize[0];
	orgImgY = origintSize[1];
	orgImgZ = origintSize[2];
	//assert(origintSize[0] == imgX); // orgImgX
	//assert(origintSize[1] == imgY); // orgImgY
	//assert(origintSize[2] == imgZ); // orgImgZ	


	if (origintSize[0] == imgX && origintSize[1] == imgY && origintSize[2] == imgZ) {
		return rescaler->GetOutput();
	}
	else {
		// 定义目标大小
		ImageType3F::SizeType targetSize;
		targetSize[0] = imgX;
		targetSize[1] = imgY;
		targetSize[2] = imgZ;

		ImageType3F::Pointer resampledImg = resizeImg(rescaler->GetOutput(), targetSize);
		return resampledImg;
	}	

	
}




/************************构建TensorRT Engine*********************************/
ICudaEngine* buildEngine(const string& engine_model_path, const string& onnx_model_path, ILogger& logger, int& EngineLoadflag)
{
	ICudaEngine *engine;
	// 判断是否存在序列化文件
	ifstream engineFile(engine_model_path, ios_base::in | ios::binary);
	if (!engineFile) {
		// miss engine file, rebuild it with the onnx file.
		ifstream onnxFile(onnx_model_path, ios_base::in | ios::binary);
		if (onnxFile) {

			try {
				// 如果不存在.engine文件则启动序列化过程，生成.engine文件，并反序列化创建engine

				IBuilder *builder = createInferBuilder(logger);
				const uint32_t explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
				INetworkDefinition *network = builder->createNetworkV2(explicit_batch);

				IParser *parser = createParser(*network, logger);
				parser->parseFromFile(onnx_model_path.c_str(), static_cast<int>(ILogger::Severity::kERROR));
				for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
					cout << parser->getError(i)->desc() << endl;
				}

				IBuilderConfig *config = builder->createBuilderConfig();
				config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 32);
				if (builder->platformHasFastFp16()) {
					config->setFlag(BuilderFlag::kFP16);
				}

				IHostMemory *serialized_model = builder->buildSerializedNetwork(*network, *config);

				if (serialized_model) {
					// 将模型序列化到engine文件中
					stringstream engine_file_stream;
					engine_file_stream.seekg(0, engine_file_stream.beg);
					engine_file_stream.write(static_cast<const char *>(serialized_model->data()), serialized_model->size());
					ofstream out_file(engine_model_path, ios_base::out | ios::binary);
					assert(out_file.is_open());
					out_file << engine_file_stream.rdbuf();
					out_file.close();

					IRuntime *runtime = createInferRuntime(logger);
					engine = runtime->deserializeCudaEngine(serialized_model->data(), serialized_model->size());

					delete config;
					delete parser;
					delete network;
					delete builder;
					delete serialized_model;
					delete runtime;
				}
				else {

					EngineLoadflag = 0;
					engine = nullptr;
				}


			}
			catch (...) {

				// worst case, engine onnx not work, failed to initialized the network
				EngineLoadflag = 0;
				engine = nullptr;
				// Initialize the iso segmentation in outer processes.
			}
			

		}
		else {
			EngineLoadflag = 0;
			engine = nullptr;
		}

		
	}
	else {

		try {
			// 如果有.engine文件，则直接读取文件，反序列化生成engine
			engineFile.seekg(0, ios::end);
			size_t engineSize = engineFile.tellg();
			engineFile.seekg(0, ios::beg);
			vector<char> engineData(engineSize);
			engineFile.read(engineData.data(), engineSize);
			engineFile.close();

			IRuntime *runtime = createInferRuntime(logger);
			assert(runtime != nullptr);
			engine = runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr);
			assert(engine != nullptr);

			delete runtime;
		}
		catch(...) {

			// worst case, engine onnx not work, failed to initialized the network
			EngineLoadflag = 0;
			engine = nullptr;
			// Initialize the iso segmentation in outer processes.
		}
	}
	return engine;
}




/************************engine预测*********************************/
float* enginePredict(ICudaEngine *engine, ImageType3F::Pointer imageData)
{
	float* imageData_buffer = imageData->GetBufferPointer();
	// 获取模型输入尺寸并分配GPU内存
	void *buffers[3];
	Dims input_dim = engine->getBindingDimensions(0);
	int input_size = 1;
	for (int j = 0; j < input_dim.nbDims; ++j) {
		input_size *= input_dim.d[j];
	}
	cudaMalloc(&buffers[0], input_size * sizeof(float));

	// 获取模型输出尺寸并分配GPU内存
	Dims output_dim = engine->getBindingDimensions(1);
	int output_size = 1;
	for (int j = 0; j < output_dim.nbDims; ++j) {
		output_size *= output_dim.d[j];
	}
	cudaMalloc(&buffers[1], output_size * sizeof(float));
	cudaMalloc(&buffers[2], output_size * sizeof(float));

	// 给模型输出数据分配相应的CPU内存
	float *output_buffer = new float[output_size]();

	// 创建cuda流
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// 拷贝输入数据
	cudaMemcpyAsync(buffers[0], imageData_buffer, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);

	// 执行推理
	IExecutionContext *context = engine->createExecutionContext();
	context->enqueueV2(buffers, stream, nullptr);
	// 拷贝输出数据
	cudaMemcpyAsync(output_buffer, buffers[2], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	////////////////////////////////保存二进制文件，仅调试用/////////////////////////////////////////
	//ofstream binaryFile("output_buffer.bin", ios::binary);
	//size_t bufferSize = output_size * sizeof(float);
	//binaryFile.write(reinterpret_cast<const char*>(output_buffer), bufferSize);
	//binaryFile.close();
	////////////////////////////////保存二进制文件，仅调试用/////////////////////////////////////////

	cudaFree(buffers[0]);
	cudaFree(buffers[1]);
	cudaFree(buffers[2]);
	cudaStreamDestroy(stream);
	context->destroy();

	return output_buffer;
}




int offsetcal(double* normvetors, int distance) {
	// new coord vector
	double vectorA[3] = { normvetors[0], normvetors[1],normvetors[2] };
	// origin vtk coord label
	double vectorB[3] = { normvetors[3], normvetors[4],normvetors[5] };

	// 计算点积
	double dotProduct = vtkMath::Dot(vectorA, vectorB);

	// 计算向量长度
	double lengthA = vtkMath::Norm(vectorA);
	double lengthB = vtkMath::Norm(vectorB);

	// 计算夹角（弧度）
	double angleRadians = std::acos(dotProduct / (lengthA * lengthB));
	
	double angleDegree = angleRadians * 180 / 3.1415926;

	double temp = cos(angleRadians)*distance;

	return int(round(cos(angleRadians)*distance));
}
int offsetcal_normv(double* normvetors, int distance) {
	// new coord vector
	double vectorA[3] = { normvetors[0], normvetors[1],normvetors[2] };
	// origin vtk coord label
	double vectorB[3] = { normvetors[3], normvetors[4],normvetors[5] };

	// 计算点积
	double dotProduct = vtkMath::Dot(vectorA, vectorB);

	// 计算向量长度
	double lengthA = vtkMath::Norm(vectorA);
	double lengthB = vtkMath::Norm(vectorB);

	// 计算夹角（弧度）
	double angleRadians = std::acos(dotProduct / (lengthA * lengthB));

	double angleDegree = angleRadians * 180 / 3.1415926;

	double temp = sin(angleRadians)*distance;
	if(angleDegree>90)
		return -int(round(sin(angleRadians)*distance));
	else
		return int(round(sin(angleRadians)*distance));
}




/*
函数说明：返回切片任意点的病人坐标系坐标，用于保存至dicom meta

const array<float, 3> oriSpacing：vtk3d 数据中的pixelspacing
const array<float, 3> vtkOrigin：vtk3d 数据中最初设置的原点坐标
double* offsets：距离切片原点的偏移量
*/
array<float, 3> getReslicePtcoord(vtkSmartPointer<vtkImageReslice> reslice, const array<float, 3> oriSpacing,
	const array<float, 3> vtkOrigin, double* offsets, int sliceflag, int sliceDirection)
{
	/////////////////////
	vtkImageData* vtkImage_reslice = reslice->GetOutput();
	double resliceOrigin[3];
	reslice->GetResliceAxesOrigin(resliceOrigin);
	double imgSpacing[3];
	vtkImage_reslice->GetSpacing(imgSpacing);
	double imgOrigin[3];
	vtkImage_reslice->GetOrigin(imgOrigin);
	int imgDimensions[3];
	vtkImage_reslice->GetDimensions(imgDimensions);


	double centerX = (imgOrigin[0] + offsets[0] * imgSpacing[0])/ imgSpacing[0];
	double centerY = (imgOrigin[1] + offsets[1] * imgSpacing[1])/ imgSpacing[1];


	double resliceDirectionX[3];
	double resliceDirectionY[3];
	double resliceDirectionZ[3];
	reslice->GetResliceAxesDirectionCosines(resliceDirectionX, resliceDirectionY, resliceDirectionZ);

	array<float, 3> ptCoordWorld;
	
	ptCoordWorld[0] = resliceOrigin[0] + centerX * imgSpacing[0] * resliceDirectionX[0] + centerY * imgSpacing[1] * resliceDirectionY[0];
	ptCoordWorld[1] = resliceOrigin[1] + centerX * imgSpacing[0] * resliceDirectionX[1] + centerY * imgSpacing[1] * resliceDirectionY[1];
	ptCoordWorld[2] = resliceOrigin[2] + centerX * imgSpacing[0] * resliceDirectionX[2] + centerY * imgSpacing[1] * resliceDirectionY[2];



	float xSliceD = centerX * imgSpacing[0] * resliceDirectionX[0] + centerY * imgSpacing[1] * resliceDirectionY[0];
	float ySliceD = centerX * imgSpacing[0] * resliceDirectionX[1] + centerY * imgSpacing[1] * resliceDirectionY[1];
	float zSliceD = centerX * imgSpacing[0] * resliceDirectionX[2] + centerY * imgSpacing[1] * resliceDirectionY[2];


	array<float, 3> ptCoordVTK = PhysicalPointToImage(ptCoordWorld, vtkOrigin, oriSpacing);
	return ImageToPatientCoord(ptCoordVTK, vtkOrigin, oriSpacing, sliceDirection);
}





/*
函数说明：利用eigen库求欧拉角

double* normvectors：长度为6，输入需要求角的两个向量
double* eulerAngles：返回指定顺序为yxz的欧拉角
int directionFlag: 0代表横断位，1为矢状位，2为冠状位
*/
int eulerAngleCal_Eigen(double* normvectors, double* eulerAngles, int directionFlag=0) {
	double pi = 3.1415926;
	// origin segmentation vector
	double vectorA[3] = { normvectors[3], normvectors[4],normvectors[5] };
	// new vector
	double vectorB[3] = { normvectors[0], normvectors[1],normvectors[2] };

	// 计算点积
	double dotProduct = vtkMath::Dot(vectorB, vectorA);

	// 计算向量长度
	double lengthA = vtkMath::Norm(vectorA);
	double lengthB = vtkMath::Norm(vectorB);

	// 计算夹角（弧度）
	double angleRadians = std::acos(dotProduct / (lengthA * lengthB));

	// 计算旋转轴
	double crossProduct1[3];
	vtkMath::Cross(vectorA, vectorB, crossProduct1);
	double len = vtkMath::Norm(crossProduct1);
	for (int i = 0; i < 3; i++) {
		crossProduct1[i] = crossProduct1[i] / len;
	}

	// 调用eigen库求指定方向欧拉角
	Eigen::AngleAxisd rotation_vector(angleRadians, Eigen::Vector3d(crossProduct1[0], crossProduct1[1], crossProduct1[2]));
	Eigen::Matrix3d rotation_matrix;
	rotation_matrix = rotation_vector.toRotationMatrix();
	// 指定欧拉角方向，0为x轴，2为z轴
	Eigen::Vector3d eulerAngle = rotation_matrix.eulerAngles(1, 0, 2);

	eulerAngles[1] = eulerAngle[0];
	eulerAngles[0] = eulerAngle[1];
	eulerAngles[2] = eulerAngle[2];
	// 由于只要锐角的欧拉角，所以以pi为单位做等效转换
	for (int i = 0; i < 3; i++) {
		eulerAngles[i] *= 180 / pi;

		if ((abs(eulerAngles[i]) < 45) && (directionFlag == 2) && (i==0)) {
			eulerAngles[i] *= -1;
		}

		if (eulerAngles[i] > 90) {
			eulerAngles[i] -= 180;
			if((directionFlag ==0) && (i==0))
				eulerAngles[i] *= -1;

		}
		else if (eulerAngles[i] < -90) {
			eulerAngles[i] += 180;
			if ((directionFlag ==0) && (i == 0))
				eulerAngles[i] *= -1;
		}

		if (eulerAngles[i] > 45)
			eulerAngles[i] = 90- eulerAngles[i];
		else if (eulerAngles[i] < -45)
			eulerAngles[i] = 90 + eulerAngles[i];

	}

	return 0;
}




/*
函数说明：填写dicomtag中特殊坐标点的病人坐标与切片的轴向坐标

double* axialDirectionCosines：切片旋转矩阵
array<float, 3> vtkOrigin：vtk数据的origin原点
const array<float, 3> oriSpacing：vtk 3d数据的原pixelspacing
vtkSmartPointer<vtkImageReslice> reslice: vtk切片
itk::MetaDataDictionary& metaData_saved：dicom meta数据
int instanceNumeber: 标记位，用于保存dicom坐标轴区分使用
*/
 
void metafilledpts(double* axialDirectionCosines, array<float, 3> vtkOrigin, const array<float, 3> oriSpacing,
	vtkSmartPointer<vtkImageReslice> reslice, itk::MetaDataDictionary& metaData_saved, int instanceNumeber,int sliceDirection) {

	// 用于拼接字符串临时变量
	std::ostringstream value;
	value.str(""); // clear, or inital

	// use instanceNumber to figure out which sclice direction we cutted here.
	if (instanceNumeber < 4) {
		// axial scice, Resliceorigin xy -> z,-x in vtk3d coord
		// save the x,y,z label direction of this slice in anke scan coord

		/*if (sliceDirection) {
			for (int cid = 0; cid < 3; cid++)
				axialDirectionCosines[cid] *= -1;
		}

		value << std::setiosflags(std::ios::fixed) << std::setprecision(5)
			<< axialDirectionCosines[2] << "\\" << -axialDirectionCosines[0] << "\\" << axialDirectionCosines[1]
			<< "\\" << axialDirectionCosines[5] << "\\" << -axialDirectionCosines[3] << "\\" << axialDirectionCosines[4]
			<< "\\" << axialDirectionCosines[8] << "\\" << -axialDirectionCosines[6] << "\\" << axialDirectionCosines[7];*/

		if (sliceDirection) {
			value << std::setiosflags(std::ios::fixed) << std::setprecision(5)
				<< -axialDirectionCosines[2] << "\\" << -axialDirectionCosines[0] << "\\" << axialDirectionCosines[1]
				<< "\\" << -axialDirectionCosines[5] << "\\" << -axialDirectionCosines[3] << "\\" << axialDirectionCosines[4]
				<< "\\" << -axialDirectionCosines[8] << "\\" << -axialDirectionCosines[6] << "\\" << axialDirectionCosines[7];
		}
		else {
			value << std::setiosflags(std::ios::fixed) << std::setprecision(5)
				<< axialDirectionCosines[2] << "\\" << -axialDirectionCosines[0] << "\\" << axialDirectionCosines[1]
				<< "\\" << axialDirectionCosines[5] << "\\" << -axialDirectionCosines[3] << "\\" << axialDirectionCosines[4]
				<< "\\" << axialDirectionCosines[8] << "\\" << -axialDirectionCosines[6] << "\\" << axialDirectionCosines[7];
		}
		

		itk::EncapsulateMetaData<string>(metaData_saved, "0065|0013", value.str());

		//save the x,y label direction of this slice in patient coord: ImageOrientationPatient	
		value.str("");
		if (sliceDirection) {
			value << std::setiosflags(std::ios::fixed) << std::setprecision(6)
				<< -axialDirectionCosines[2] << "\\" << axialDirectionCosines[0] << "\\" << -axialDirectionCosines[1]
				<< "\\" << -axialDirectionCosines[5] << "\\" << axialDirectionCosines[3] << "\\" << -axialDirectionCosines[4];
			itk::EncapsulateMetaData<string>(metaData_saved, "0020|0037", value.str());
		}
		else {
			value << std::setiosflags(std::ios::fixed) << std::setprecision(6)
				<< axialDirectionCosines[2] << "\\" << -axialDirectionCosines[0] << "\\" << axialDirectionCosines[1]
				<< "\\" << -axialDirectionCosines[5] << "\\" << axialDirectionCosines[3] << "\\" << -axialDirectionCosines[4];
			itk::EncapsulateMetaData<string>(metaData_saved, "0020|0037", value.str());
		}
		
	}
	else if (instanceNumeber < 7) {
		// sagital scice, Resliceorigin xy -> x,-y in vtk3d coord
		// save the x,y,z label direction of this slice in anke scan coord

		if (sliceDirection) {
			value << std::setiosflags(std::ios::fixed) << std::setprecision(5)
				<< -axialDirectionCosines[2] << "\\" << -axialDirectionCosines[0] << "\\" << axialDirectionCosines[1]
				<< "\\" << -axialDirectionCosines[5] << "\\" << -axialDirectionCosines[3] << "\\" << axialDirectionCosines[4]
				<< "\\" << axialDirectionCosines[8] << "\\" << -axialDirectionCosines[6] << "\\" << axialDirectionCosines[7];			
			
		}
		else
		{
			// left side orientation
			for (int cid = 6; cid < 9; cid++)
				axialDirectionCosines[cid] *= -1;
			value << std::setiosflags(std::ios::fixed) << std::setprecision(5)
			<< axialDirectionCosines[2] << "\\" << -axialDirectionCosines[0] << "\\" << axialDirectionCosines[1]
			<< "\\" << axialDirectionCosines[5] << "\\" << -axialDirectionCosines[3] << "\\" << axialDirectionCosines[4]
			<< "\\" << axialDirectionCosines[8] << "\\" << -axialDirectionCosines[6] << "\\" << axialDirectionCosines[7];

		}	
		itk::EncapsulateMetaData<string>(metaData_saved, "0065|0013", value.str());

		//save the x,y label direction of this slice in patient coord: ImageOrientationPatient	
		value.str("");
		value << std::setiosflags(std::ios::fixed) << std::setprecision(6)
			<< -axialDirectionCosines[2] << "\\" << axialDirectionCosines[0] << "\\" << -axialDirectionCosines[1]
			<< "\\" << -axialDirectionCosines[5] << "\\" << axialDirectionCosines[3] << "\\" << -axialDirectionCosines[4];
		itk::EncapsulateMetaData<string>(metaData_saved, "0020|0037", value.str());
	}
	else {
		// coro slice
		// save the x,y,z label direction of this slice in anke scan coord

		/*if (sliceDirection) {
			for (int cid = 0; cid < 3; cid++)
				axialDirectionCosines[cid] *= -1;
		}*/


		if (sliceDirection) {

			value << std::setiosflags(std::ios::fixed) << std::setprecision(5)
				<< -axialDirectionCosines[2] << "\\" << -axialDirectionCosines[0] << "\\" << axialDirectionCosines[1]
				<< "\\" << -axialDirectionCosines[5] << "\\" << -axialDirectionCosines[3] << "\\" << axialDirectionCosines[4]
				<< "\\" << -axialDirectionCosines[8] << "\\" << -axialDirectionCosines[6] << "\\" << axialDirectionCosines[7];
			itk::EncapsulateMetaData<string>(metaData_saved, "0065|0013", value.str());


			//save the x,y label direction of this slice in patient coord: ImageOrientationPatient	
			value.str("");
			value << std::setiosflags(std::ios::fixed) << std::setprecision(6)
				<< -axialDirectionCosines[2] << "\\" << axialDirectionCosines[0] << "\\" << -axialDirectionCosines[1]
				<< "\\" << -axialDirectionCosines[5] << "\\" << axialDirectionCosines[3] << "\\" << -axialDirectionCosines[4];
			itk::EncapsulateMetaData<string>(metaData_saved, "0020|0037", value.str());



		}
		else {
			value << std::setiosflags(std::ios::fixed) << std::setprecision(5)
				<< axialDirectionCosines[2] << "\\" << -axialDirectionCosines[0] << "\\" << axialDirectionCosines[1]
				<< "\\" << axialDirectionCosines[5] << "\\" << -axialDirectionCosines[3] << "\\" << axialDirectionCosines[4]
				<< "\\" << axialDirectionCosines[8] << "\\" << -axialDirectionCosines[6] << "\\" << axialDirectionCosines[7];
			itk::EncapsulateMetaData<string>(metaData_saved, "0065|0013", value.str());


			//save the x,y label direction of this slice in patient coord: ImageOrientationPatient	
			value.str("");
			value << std::setiosflags(std::ios::fixed) << std::setprecision(6)
				<< axialDirectionCosines[2] << "\\" << -axialDirectionCosines[0] << "\\" << axialDirectionCosines[1]
				<< "\\" << -axialDirectionCosines[5] << "\\" << axialDirectionCosines[3] << "\\" << -axialDirectionCosines[4];
			itk::EncapsulateMetaData<string>(metaData_saved, "0020|0037", value.str());
		}



	}

	// now we could obtain the key cheating corner pts coordinate & save in meta
	array<float, 3> LeftLowerPoint; array<float, 3> LeftUpperPoint;
	array<float, 3> centerPoint; array<float, 3> RightUpperPoint;

	/*double offsets[2] = {0}; int imgDimensions[3];
	reslice->GetOutput()->GetDimensions(imgDimensions);
	LeftLowerPoint = getReslicePtcoord(reslice, oriSpacing, vtkOrigin, offsets, instanceNumeber);
	offsets[1] = double(imgDimensions[1]); 
	LeftUpperPoint = getReslicePtcoord(reslice, oriSpacing, vtkOrigin, offsets, instanceNumeber);
	offsets[0] = double(imgDimensions[0]);
	RightUpperPoint = getReslicePtcoord(reslice, oriSpacing, vtkOrigin, offsets, instanceNumeber);
	offsets[1] = 0; 
	RightLowerPoint = getReslicePtcoord(reslice, oriSpacing, vtkOrigin, offsets, instanceNumeber);*/


	double offsets[2] = { 0 }; int imgDimensions[3];
	reslice->GetOutput()->GetDimensions(imgDimensions);
	double imgOrigin[3];
	reslice->GetOutput()->GetOrigin(imgOrigin);
	double imgSpacing[3];
	reslice->GetOutput()->GetSpacing(imgSpacing);
	//if (instanceNumeber < 7) {
	//	// for coro & trans slice, pt calculate would be little different
	//	RightUpperPoint = getReslicePtcoord(reslice, oriSpacing, vtkOrigin, offsets, instanceNumeber, sliceDirection);
	//	offsets[0] = double(imgDimensions[0]);
	//	LeftUpperPoint = getReslicePtcoord(reslice, oriSpacing, vtkOrigin, offsets, instanceNumeber, sliceDirection);
	//	offsets[1] = double(imgDimensions[1]);
	//	LeftLowerPoint = getReslicePtcoord(reslice, oriSpacing, vtkOrigin, offsets, instanceNumeber, sliceDirection);
	//}
	//else {
		LeftUpperPoint = getReslicePtcoord(reslice, oriSpacing, vtkOrigin, offsets, instanceNumeber, sliceDirection);
		offsets[1] = double(imgDimensions[1]);// *imgSpacing[1];// double(imgDimensions[1]);
		LeftLowerPoint = getReslicePtcoord(reslice, oriSpacing, vtkOrigin, offsets, instanceNumeber, sliceDirection);
		offsets[0] = double(imgDimensions[0]);// *imgSpacing[0];
		offsets[1] = 0;
		RightUpperPoint = getReslicePtcoord(reslice, oriSpacing, vtkOrigin, offsets, instanceNumeber, sliceDirection);
	//}
	

	//offsets[0] = double(imgDimensions[0]) / 2.0; //* imgSpacing[0]
	//offsets[1] = double(imgDimensions[1]) /2.0; // * imgSpacing[1]

	// for method1
	offsets[0] = double(-imgOrigin[0]) / imgSpacing[0];
	offsets[1] = double(-imgOrigin[1]) / imgSpacing[1];
	centerPoint=getReslicePtcoord(reslice, oriSpacing, vtkOrigin, offsets, instanceNumeber, sliceDirection);


	
	// save ImagePositionPatient
	value.str(""); value << int(round(LeftUpperPoint[0])) << "\\" << int(round(LeftUpperPoint[1]))
		<< "\\" << int(round(LeftUpperPoint[2]));
	itk::EncapsulateMetaData<string>(metaData_saved, "0020|0032", value.str());
	// Privatetag, cheating pt in anke scan?
	value.str(""); value << LeftUpperPoint[0]<< "\\" << -LeftUpperPoint[1] << "\\" <<
		-LeftUpperPoint[2]<< "\\" << RightUpperPoint[0]<< "\\" <<-RightUpperPoint[1]<<
		"\\"<< -RightUpperPoint[2]<< "\\" << centerPoint[0] << "\\"<<
		-centerPoint[1] << "\\"<< -centerPoint[2]<<"\\" << LeftLowerPoint[0] << "\\" << -LeftLowerPoint[1]
		<< "\\" << -LeftLowerPoint[2];
	itk::EncapsulateMetaData<string>(metaData_saved, "0065|0012", value.str());

	
}



// 转换输出点的格式
vector<array<float, 3>> ptint2float(vector<array<int, 3>> out_points){
	vector<array<float, 3>> out_points_float;
	for (int ptid = 0; ptid < out_points.size(); ptid++) {
		array<float, 3> tempPt; tempPt[0] = float(out_points[ptid][0]);
		tempPt[1] = float(out_points[ptid][1]);
		tempPt[2] = float(out_points[ptid][2]);
		out_points_float.push_back(tempPt);
	}
	return out_points_float;
}




/*
Function: apexScanParaSet

Descirption: Use for calculate the ofs & obs of apex later scanning need.
@param metaori: Contains metadata for later save in dicom
@param patientPts: contain 5 feature pts patient location 
@param normalVectors: contain norm vectors of axial, sagittal, coronal
@param offsets: save offset of scan slice (ordered by axial, sagittal, coronal)
@param eulerAngles: save angles of scan slice (same as former)
@param sliceDirection: flag of scan direction
*/
void apexScanParaSet(itk::MetaDataDictionary& metaori, vector<array<float, 3>> patientPts,
	vector<array<float, 3>> normalVectors,double* offsets, double* eulerAngles, int sliceDirection) {

	offsets[0] = 0.5*(patientPts[3][0] + patientPts[4][0]);
	for (int i = 1; i < 3; i++)
		offsets[i] = -0.5*(patientPts[3][i] + patientPts[4][i]);

	// 记录横断面与标准扫描面横断面在anke扫描坐标系下的欧拉角
	double normvectors[6] = { 0 }; normvectors[5] = -1;
	if (sliceDirection) 
		normvectors[0] = -normalVectors[0][2];		
	else 
		normvectors[0] = normalVectors[0][2];

	normvectors[1] = -normalVectors[0][0]; normvectors[2] = normalVectors[0][1];
	eulerAngleCal_Eigen(normvectors, eulerAngles,0);
	
	// Apex test
	if (sliceDirection) {
		for (int id = 0; id < 3; id++)
			eulerAngles[id] = -eulerAngles[id];
	}
	else {
		for (int id = 0; id < 3; id++)
			eulerAngles[id] = -eulerAngles[id];
	}
	
	eulerAngles[1] = -eulerAngles[1];


	// 记录矢状面与标准扫描面的欧拉角
	normvectors[5] = 0; 
	if (sliceDirection) {
		normvectors[0] = -normalVectors[1][2];
		normvectors[3] = -1;
	}
	else {
		normvectors[0] = normalVectors[1][2];
		normvectors[3] = 1;
	}
	normvectors[1] = -normalVectors[1][0]; normvectors[2] = normalVectors[1][1];

	double eulerAngleTemp[3];
	eulerAngleCal_Eigen(normvectors, eulerAngleTemp,1);

	// Apex test
	//if (sliceDirection) {
		for (int i = 3; i < 6; i++)
			eulerAngles[i] = -eulerAngleTemp[i - 3];
	//}

	

	// 记录对应切面偏移量，由于是矢状面，根据扫描fov的经验，点2投影至矢状面上的点位置坐标貌似最好
	// 这里先尽快上线于是暂时用点2代替
	offsets[3] = (patientPts[3][0] + patientPts[4][0] + patientPts[2][0]) / 3;
	for (int i = 4; i < 6; i++)
		offsets[i] = -1 * (patientPts[3][i - 3] + patientPts[4][i - 3] + patientPts[2][i - 3]) / 3;


	// 记录冠状面与标准冠状扫描面的欧拉角
	normvectors[3] = 0; 

	if (sliceDirection) {
		normvectors[4] = -1;

		normvectors[0] = normalVectors[2][2];
		normvectors[1] = normalVectors[2][0]; normvectors[2] = -normalVectors[2][1];
	}
	else {
		normvectors[4] = 1;

		normvectors[0] = normalVectors[2][2];
		normvectors[1] = -normalVectors[2][0]; normvectors[2] = normalVectors[2][1];
	}

	eulerAngleCal_Eigen(normvectors, eulerAngleTemp,2);

	// Test by Apex window
	eulerAngles[6] = eulerAngleTemp[0];
	for (int i = 7; i < 9; i++)
		eulerAngles[i] = -eulerAngleTemp[i - 6];


	// For experienced test, the euler angle cal in eigen would lost one axial for
	// Apex need, so we need the par from other two slice direction
	if(eulerAngles[5]<0)
		eulerAngles[2] = eulerAngles[5] - 1;
	else
		eulerAngles[2] = eulerAngles[5] + 1;

	eulerAngles[3] = eulerAngles[0];
	eulerAngles[1] = eulerAngles[4];
	
	eulerAngles[7] = eulerAngles[4];
	
	

	// 记录对应切面偏移量，由于是冠状面，暂使用该点23的中点作为切片中心			
	/*offsets[6] = 0.5*(patientPts[4][0] + patientPts[2][0]);
	for (int i = 7; i < 9; i++)
		offsets[i] = -0.5*(patientPts[4][i - 6] + patientPts[2][i - 6]);*/
	offsets[6] = patientPts[1][0];
	for (int i = 7; i < 9; i++)
		offsets[i] = -patientPts[1][i - 6];

	// save eulerangles and offset to specific dicom tag
	std::ostringstream value;
	value.str(""); 
	value << std::setiosflags(std::ios::fixed) << std::setprecision(3);
	
	// value << "angles" << "\\";
	for (int id = 0; id < 9; id++)
		value << eulerAngles[id] << "\\";
	// value << "offsets" << "\\";
	for (int id = 0; id < 9; id++)
		value << offsets[id] << "\\";

	// 填入协商好的记录位置
	itk::EncapsulateMetaData<string>(metaori, "0065|0036", value.str());
}




/************************MPR输出切面*********************************
Function: mprProcess

Descirption: Segment the output mpr localizer dcms with pts cal by ai model.
@param normalVectors: contain norm vectors of axial, sagittal, coronal
@param metaori: Contains metadata for later save in dicom
@param out_points: contain 5 feature pts output form ai model
@param offsets: save offset of scan slice (ordered by axial, sagittal, coronal)
@param eulerAngles: save angles of scan slice (same as former)
@param savedcmPath: folder use for saving the 9 mpr dcms
*/
void mprProcess(ImageType3F::Pointer itk_image, vector<array<int, 3>> out_points, 
	itk::MetaDataDictionary metaori, double* offsets, double* eulerAngles, string savedcmPath)
{
	// itk转化为vtk数据
	using ITKToVTKFilterType = itk::ImageToVTKImageFilter<ImageType3F>;
	ITKToVTKFilterType::Pointer itkToVtkFilter = ITKToVTKFilterType::New();
	itkToVtkFilter->SetInput(itk_image);
	itkToVtkFilter->Update();
	vtkImageData* vtkImage = itkToVtkFilter->GetOutput();

	int dims[3];
	vtkImage->GetDimensions(dims);


	array<float, 3> origin = {
		static_cast<float>(vtkImage->GetOrigin()[0]),
		static_cast<float>(vtkImage->GetOrigin()[1]),
		static_cast<float>(vtkImage->GetOrigin()[2])
	};

	array<float, 3> origin_vtk = origin;


	array<float, 3> spacing = {
		static_cast<float>(vtkImage->GetSpacing()[0]),
		static_cast<float>(vtkImage->GetSpacing()[1]),
		static_cast<float>(vtkImage->GetSpacing()[2])
	};


	// 用于vtk切片生成的坐标
	array<float, 3> point1 = ImageToPhysicalPoint(out_points[0], origin_vtk, spacing);
	array<float, 3> point2 = ImageToPhysicalPoint(out_points[1], origin_vtk, spacing);
	array<float, 3> point3 = ImageToPhysicalPoint(out_points[2], origin_vtk, spacing);
	array<float, 3> point4 = ImageToPhysicalPoint(out_points[3], origin_vtk, spacing);
	array<float, 3> point5 = ImageToPhysicalPoint(out_points[4], origin_vtk, spacing);
	
	
	// 判定序列扫描方向，会影响到vtk的坐标系生成换算规则
	int sliceDirection = 1;
	if (origin_vtk[0] < 0)
		sliceDirection = 0;

	
	// 类型转换，用于计算扫描切片的obs,ofs
	vector<array<float, 3>> out_points_float = ptint2float(out_points);
	// 计算病人坐标系坐标, 以用于apex扫描定位
	for(int ptid=0;ptid<5;ptid++)
		out_points_float[ptid] = ImageToPatientCoord(out_points_float[ptid], origin_vtk, spacing, sliceDirection);


	array<float, 3> v1_sagittal;
	array<float, 3> v2_sagittal;
	array<float, 3> v1_axial;
	array<float, 3> v2_axial;
	array<float, 3> v1_coronal;
	array<float, 3> v2_coronal;
	array<float, 3> normal_sagittal;
	array<float, 3> normal_axial;
	array<float, 3> normal_coronal;

	// 计算矢状面的法线向量
	SubtractVectors(point5, point1, v1_sagittal);
	NormalizeVector(v1_sagittal);
	SubtractVectors(point3, point1, v2_sagittal);
	NormalizeVector(v2_sagittal);	
	normal_sagittal = cal_norm_vec(v1_sagittal, v2_sagittal);
	// 更新v2, 用于后期生成新的切片使用
	SubtractVectors(point5, point3, v2_sagittal);
	NormalizeVector(v2_sagittal);

	// 计算横断面的法线向量
	SubtractVectors(point5, point4, v1_axial);
	NormalizeVector(v1_axial);
	v2_axial = normal_sagittal;
	normal_axial = cal_norm_vec(v1_axial, v2_axial);

	// 计算冠状面的法线向量
	SubtractVectors(point2, point3, v1_coronal);
	NormalizeVector(v1_coronal);
	v2_coronal = normal_sagittal;
	normal_coronal = cal_norm_vec(v1_coronal, v2_coronal);


	// 优先计算所有扫描面的欧拉角以及对应偏移量填入meta
	vector<array<float, 3>> normalVectors;
	normalVectors.push_back(normal_axial);
	normalVectors.push_back(normal_sagittal);
	normalVectors.push_back(normal_coronal);
	apexScanParaSet(metaori, out_points_float, normalVectors,
		offsets, eulerAngles, sliceDirection);



	/* 
	   因为工作流设定，一次性完成所有结果保存，根据重建流程定义路径文件夹保存9个切片图像
	   切片有顺序要求： 3轴3矢3冠；p.s. 排序在(0020,0013)，从1开始
	*/


	// 创建Dcm图像IO对象
	/* 经过实验发现itk关于study & series Id 是与dicomI0直接绑定的，
	不能人为修改这个数值，所以提前创建一个公用IO即可使得接下来所有
	输出图像均为一个“定位相序列”*/
	itk::GDCMImageIO::Pointer dcmIO = itk::GDCMImageIO::New();
	
	// 创建临时string变量, 用于更新对应dcm的保存路径名
	std::ostringstream value; value.str("");

	
	// 或许序列号为后面的切片名字保存做准备
	// metaori.GetKeys(); //?
	dcmIO->SetMetaDataDictionary(metaori);
	string seriesNumstr; dcmIO->GetValueFromTag("0020|0011", seriesNumstr);
	int seriesNum = atoi(seriesNumstr.c_str());
	seriesNum *= 10000;
	int sliceid = 0;

	// 开始依次生成对应扫描切片并保存
	// 切片出片逻辑修改为常规定位出片逻辑，即
	// 1. 生成横断切片为3片等距（常规定位像层间隔10mm）,同方向；
	// 2. 矢、冠同理...

	int sliceGap = 20;

	//for (int mode = 0; mode < 3; mode++) {
	//	// 针对三种切片方向做判定生成三种不同的轴向计算公式即可，然后通过循环
	//	// 计算三张相邻切片
	//	for (int imgid = 1; imgid < 3; imgid++) {

	//	}
	//}
	//



	// 1. MPR得到横断面
	array<float, 3> axis1_axial = v1_axial;
	array<float, 3> axis2_axial = normal_axial;
	// 通过此时axis1,2获得新的axis3，确保三轴垂直
	array<float, 3> axis3_axial = cal_norm_vec(axis1_axial, axis2_axial);//v2_axial;

	// 设置切片中心
	double origin_cut[3] = { 0.5*(point4[0] + point5[0]),
		0.5*(point4[1] + point5[1]),
		0.5*(point4[2] + point5[2]) };

	if (sliceDirection) {
		for (int cid = 0; cid < 3; cid++)
			axis3_axial[cid] *= -1;
	}

	
	vtkSmartPointer<vtkImageReslice> resliceAxialAtAxial = vtkSmartPointer<vtkImageReslice>::New();
	resliceAxialAtAxial->SetInputData(vtkImage);
	resliceAxialAtAxial->SetOutputDimensionality(2);

	// 设置切片平面的方向和原点
	double axialDirectionCosines[9] = { -axis3_axial[0], -axis3_axial[1], -axis3_axial[2],
										axis1_axial[0], axis1_axial[1], axis1_axial[2],
										axis2_axial[0], axis2_axial[1], axis2_axial[2] };


	resliceAxialAtAxial->SetResliceAxesDirectionCosines(axialDirectionCosines); 
	resliceAxialAtAxial->SetResliceAxesOrigin(origin_cut);
	resliceAxialAtAxial->SetInterpolationModeToLinear();
	resliceAxialAtAxial->Update();


	// 更新部分meta信息
	sliceid = 2;
	metafilledpts(axialDirectionCosines, origin_vtk, spacing,
		resliceAxialAtAxial, metaori, sliceid, sliceDirection);

	// 保存当前切面dcm
	vtkImageData* vtkImage_resliceAxial = resliceAxialAtAxial->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid-1) << ".dcm";
	SaveAsDCM(vtkImage_resliceAxial, metaori, dcmIO, value.str(), sliceid);


	// 生成相邻轴位切片，仅改变中心点origin_cut的位置
	// 未来加上角度计算，暂时只利用vtk-y计算
	origin_cut[1] = origin_cut[1] - sliceGap;
	resliceAxialAtAxial->SetResliceAxesOrigin(origin_cut);
	resliceAxialAtAxial->Update();
	sliceid = 1;
	metafilledpts(axialDirectionCosines, origin_vtk, spacing,
		resliceAxialAtAxial, metaori, sliceid, sliceDirection);

	// 保存当前切面dcm
	vtkImage_resliceAxial = resliceAxialAtAxial->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid-1) << ".dcm";
	SaveAsDCM(vtkImage_resliceAxial, metaori, dcmIO, value.str(), sliceid);


	// 未来加上角度计算，暂时只利用vtk-y计算
	origin_cut[1] = origin_cut[1] + 2* sliceGap;
	resliceAxialAtAxial->SetResliceAxesOrigin(origin_cut);
	resliceAxialAtAxial->Update();
	sliceid = 3;
	metafilledpts(axialDirectionCosines, origin_vtk, spacing,
		resliceAxialAtAxial, metaori, sliceid, sliceDirection);

	// 保存当前切面dcm
	vtkImage_resliceAxial = resliceAxialAtAxial->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceAxial, metaori, dcmIO, value.str(), sliceid);






	// 2. MPR得到矢状面
	array<float, 3> axis2_sagittal = v1_coronal;// v2_sagittal;
	array<float, 3> axis1_sagittal = cal_norm_vec(axis2_sagittal, normal_sagittal);

	for (int cid = 0; cid < 3; cid++)
		axis1_sagittal[cid] *= -1;

	array<float, 3> axis3_sagittal = normal_sagittal;
	origin_cut[0] = (point4[0] + point5[0] + point3[0]) / 3;
	origin_cut[1] = (point4[1] + point5[1] + point3[1]) / 3;
	origin_cut[2] = (point4[2] + point5[2] + point3[2]) / 3;


	if (sliceDirection) {
		for (int cid = 0; cid < 3; cid++) {
			//axis2_sagittal[cid] *= -1; //
			axis3_sagittal[cid] *= -1;
		}
	}


	vtkSmartPointer<vtkImageReslice> resliceSagittalAtSagittal = vtkSmartPointer<vtkImageReslice>::New();
	resliceSagittalAtSagittal->SetInputData(vtkImage);
	resliceSagittalAtSagittal->SetOutputDimensionality(2);

	// 设置切片平面的方向和原点
	double sagittalDirectionCosines[9] = { axis1_sagittal[0], axis1_sagittal[1], axis1_sagittal[2],
										   -axis2_sagittal[0], -axis2_sagittal[1], -axis2_sagittal[2],
										   -axis3_sagittal[0], -axis3_sagittal[1], -axis3_sagittal[2] };


	resliceSagittalAtSagittal->SetResliceAxesDirectionCosines(sagittalDirectionCosines);
	resliceSagittalAtSagittal->SetResliceAxesOrigin(origin_cut);
	resliceSagittalAtSagittal->SetInterpolationModeToLinear();
	resliceSagittalAtSagittal->Update();


	// 更新部分meta信息
	sliceid = 5;
	metafilledpts(sagittalDirectionCosines, origin_vtk, spacing,
		resliceSagittalAtSagittal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImageData* vtkImage_resliceSagittal = resliceSagittalAtSagittal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid-1) << ".dcm";
	SaveAsDCM(vtkImage_resliceSagittal, metaori, dcmIO, value.str(), sliceid);

	// 生成相邻轴位切片，仅改变中心点origin_cut的位置
	// 未来加上角度计算，暂时只利用vtk-z计算
	origin_cut[2] = origin_cut[2] - sliceGap;
	resliceSagittalAtSagittal->SetResliceAxesOrigin(origin_cut);
	resliceSagittalAtSagittal->Update();
	// 更新部分meta信息
	sliceid = 4;
	metafilledpts(sagittalDirectionCosines, origin_vtk, spacing,
		resliceSagittalAtSagittal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImage_resliceSagittal = resliceSagittalAtSagittal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceSagittal, metaori, dcmIO, value.str(), sliceid);


	// 未来加上角度计算，暂时只利用vtk - y计算
	origin_cut[2] = origin_cut[2] + sliceGap*2;
	resliceSagittalAtSagittal->SetResliceAxesOrigin(origin_cut);
	resliceSagittalAtSagittal->Update();
	// 更新部分meta信息
	sliceid = 6;
	metafilledpts(sagittalDirectionCosines, origin_vtk, spacing,
		resliceSagittalAtSagittal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImage_resliceSagittal = resliceSagittalAtSagittal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceSagittal, metaori, dcmIO, value.str(), sliceid);





	// 3. MPR得到冠状面
	array<float, 3> axis1_coronal = cal_norm_vec(v2_coronal, v1_coronal);
	array<float, 3> axis2_coronal = v1_coronal;
	// 利用已知垂直计算第三轴, 确保垂直
	array<float, 3> axis3_coronal = cal_norm_vec(axis1_coronal, axis2_coronal);

	origin_cut[0] = point2[0]; //(point5[0] + point3[0]) / 2;
	origin_cut[1] = point2[1]; //(point5[1] + point3[1]) / 2;
	origin_cut[2] = point2[2]; //(point5[2] + point3[2]) / 2;

	if (sliceDirection) {	
		for (int cid = 0; cid < 3; cid++){
			axis3_coronal[cid] *= -1;
			//axis1_coronal[cid] *= -1;
		}
	}
	else {
		/*for (int cid = 0; cid < 3; cid++) {			
			axis1_coronal[cid] *= -1;
		}*/
	}

	vtkSmartPointer<vtkImageReslice> resliceCoronalAtCoronal = vtkSmartPointer<vtkImageReslice>::New();
	resliceCoronalAtCoronal->SetInputData(vtkImage);
	resliceCoronalAtCoronal->SetOutputDimensionality(2);

	// 设置切片平面的方向和原点
	double coronalDirectionCosines[9] = { -axis3_coronal[0], -axis3_coronal[1], -axis3_coronal[2],
										  -axis2_coronal[0], -axis2_coronal[1], -axis2_coronal[2],
										  axis1_coronal[0], axis1_coronal[1], axis1_coronal[2] };


	resliceCoronalAtCoronal->SetResliceAxesDirectionCosines(coronalDirectionCosines);
	resliceCoronalAtCoronal->SetResliceAxesOrigin(origin_cut);
	resliceCoronalAtCoronal->SetInterpolationModeToLinear();
	resliceCoronalAtCoronal->Update();

	// 更新部分特殊meta信息
	sliceid = 8;
	metafilledpts(coronalDirectionCosines, origin_vtk, spacing,
		resliceCoronalAtCoronal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImageData* vtkImage_resliceCoronal = resliceCoronalAtCoronal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid-1) << ".dcm";
	SaveAsDCM(vtkImage_resliceCoronal, metaori, dcmIO, value.str(), sliceid);

	// 生成相邻轴位切片，仅改变中心点origin_cut的位置
	// 未来加上角度计算，暂时只利用vtk-x计算
	origin_cut[0] = origin_cut[0] - sliceGap;
	resliceCoronalAtCoronal->SetResliceAxesOrigin(origin_cut);
	resliceCoronalAtCoronal->Update();
	// 更新部分特殊meta信息
	sliceid = 7;
	metafilledpts(coronalDirectionCosines, origin_vtk, spacing,
		resliceCoronalAtCoronal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImage_resliceCoronal = resliceCoronalAtCoronal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceCoronal, metaori, dcmIO, value.str(), sliceid);

	// 未来加上角度计算，暂时只利用vtk-x计算
	origin_cut[0] = origin_cut[0] + sliceGap * 2;
	resliceCoronalAtCoronal->SetResliceAxesOrigin(origin_cut);
	resliceCoronalAtCoronal->Update();
	// 更新部分特殊meta信息
	sliceid = 9;
	metafilledpts(coronalDirectionCosines, origin_vtk, spacing,
		resliceCoronalAtCoronal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImage_resliceCoronal = resliceCoronalAtCoronal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceCoronal, metaori, dcmIO, value.str(), sliceid);



}




/*
Function: mprIsoCenter

Descirption: Use for MPR segament backup regular localizer.
@param itk_image: 3D itk img data; 
@param metaori: Contains metadata for later save in dicom;
@param savedcmPath: backup folder savepath.
*/
void mprIsoCenter(ImageType3F::Pointer itk_image, itk::MetaDataDictionary metaori, string savedcmPath) {

	// itk转化为vtk数据
	using ITKToVTKFilterType = itk::ImageToVTKImageFilter<ImageType3F>;
	ITKToVTKFilterType::Pointer itkToVtkFilter = ITKToVTKFilterType::New();
	itkToVtkFilter->SetInput(itk_image);
	itkToVtkFilter->Update();
	vtkImageData* vtkImage = itkToVtkFilter->GetOutput();

	int dims[3];
	vtkImage->GetDimensions(dims);


	array<float, 3> origin = {
		static_cast<float>(vtkImage->GetOrigin()[0]),
		static_cast<float>(vtkImage->GetOrigin()[1]),
		static_cast<float>(vtkImage->GetOrigin()[2])
	};

	array<float, 3> origin_vtk = origin;


	array<float, 3> spacing = {
		static_cast<float>(vtkImage->GetSpacing()[0]),
		static_cast<float>(vtkImage->GetSpacing()[1]),
		static_cast<float>(vtkImage->GetSpacing()[2])
	};

	// 判定序列扫描方向，会影响到vtk的坐标系生成换算规则
	int sliceDirection = 1;
	if (origin_vtk[0] < 0)
		sliceDirection = 0;

	

	// 创建Dcm图像IO对象
	/* 经过实验发现itk关于study & series Id 是与dicomI0直接绑定的，
	   不能人为修改这个数值，所以提前创建一个公用IO即可使得接下来所有
	   输出图像均为一个“定位相序列”*/
	itk::GDCMImageIO::Pointer dcmIO = itk::GDCMImageIO::New();

	// 创建临时string变量, 用于更新对应dcm的保存路径名
	std::ostringstream value; value.str("");


	// 优先计算所有扫描面的欧拉角以及对应偏移量填入meta
	// 由于是等中心切片，因此所有切片的偏移量、旋转角均为0
	// value << "angles" << "\\";
	for (int id = 0; id < 9; id++)
		value <<0 << "\\";
	// value << "offsets" << "\\";
	for (int id = 0; id < 9; id++)
		value <<0 << "\\";

	// 填入协商好的记录位置
	itk::EncapsulateMetaData<string>(metaori, "0065|0036", value.str());


	/*
		因为工作流设定，一次性完成所有结果保存，根据重建流程定义路径文件夹保存9个切片图像
		切片有顺序要求： 3轴3矢3冠；p.s. 排序在(0020,0013)，从1开始
	*/
	dcmIO->SetMetaDataDictionary(metaori);
	string seriesNumstr; dcmIO->GetValueFromTag("0020|0011", seriesNumstr);
	int seriesNum = atoi(seriesNumstr.c_str());
	seriesNum *= 10000;
	int sliceid = 0;

	// 开始依次生成对应扫描切片并保存
	// 切片出片逻辑修改为常规定位出片逻辑，即
	// 1. 生成横断切片为3片等距（常规定位像层间隔10mm）,同方向；
	// 2. 矢、冠同理...

	int sliceGap = 20;
	array<int, 3> imagePoint = { int(0.5 * dims[0]),
		int(0.45 * dims[1]),
		int(0.5 * dims[2])
	};
	array<float, 3> point1 = ImageToPhysicalPoint(imagePoint, origin_vtk, spacing);


	// 1. MPR得到横断面
	double origin_cut[3] = { point1[0],
		point1[1],
		point1[2] };

	vtkSmartPointer<vtkImageReslice> resliceAxialAtAxial = vtkSmartPointer<vtkImageReslice>::New();
	resliceAxialAtAxial->SetInputData(vtkImage);
	resliceAxialAtAxial->SetOutputDimensionality(2);

	double axialX[3] = { 0,0,-1 };
	double axialY[3] = { 1,0,0 };
	double axialZ[3] = { 0,-1,0 };
	if (sliceDirection) {
		for (int cid = 0; cid < 3; cid++)
			axialX[cid] *= -1;
	}

	double axialDirectionCosines[9] = { -axialX[0], -axialX[1], -axialX[2],
										axialY[0], axialY[1], axialY[2],
										axialZ[0], axialZ[1], axialZ[2] };

	resliceAxialAtAxial->SetResliceAxesDirectionCosines(axialDirectionCosines);
	resliceAxialAtAxial->SetResliceAxesOrigin(origin_cut);
	resliceAxialAtAxial->SetInterpolationModeToLinear();
	resliceAxialAtAxial->Update();

	// 更新部分meta信息
	sliceid = 2;
	metafilledpts(axialDirectionCosines, origin_vtk, spacing,
		resliceAxialAtAxial, metaori, sliceid, sliceDirection);

	// 保存当前切面dcm
	vtkImageData* vtkImage_resliceAxial = resliceAxialAtAxial->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceAxial, metaori, dcmIO, value.str(), sliceid);


	// 生成相邻轴位切片，仅改变中心点origin_cut的位置
	// 利用vtk-y计算相邻切片
	origin_cut[1] = origin_cut[1] - sliceGap;
	resliceAxialAtAxial->SetResliceAxesOrigin(origin_cut);
	resliceAxialAtAxial->Update();
	sliceid = 1;
	metafilledpts(axialDirectionCosines, origin_vtk, spacing,
		resliceAxialAtAxial, metaori, sliceid, sliceDirection);

	// 保存当前切面dcm
	vtkImage_resliceAxial = resliceAxialAtAxial->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceAxial, metaori, dcmIO, value.str(), sliceid);

	// 利用vtk-y计算计算相邻切片
	origin_cut[1] = origin_cut[1] + 2 * sliceGap;
	resliceAxialAtAxial->SetResliceAxesOrigin(origin_cut);
	resliceAxialAtAxial->Update();
	sliceid = 3;
	metafilledpts(axialDirectionCosines, origin_vtk, spacing,
		resliceAxialAtAxial, metaori, sliceid, sliceDirection);

	// 保存当前切面dcm
	vtkImage_resliceAxial = resliceAxialAtAxial->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceAxial, metaori, dcmIO, value.str(), sliceid);
	origin_cut[1] -= sliceGap;



	// 2. MPR得到矢状面
	axialX[2] = 0;
	if (sliceDirection) {
		axialX[0] = -1;
	}
	else
		axialX[0] = 1; // wired!!!!!
	axialY[0] = 0; axialY[1] = 1;
	axialZ[2] = -1; axialZ[1] = 0;

	vtkSmartPointer<vtkImageReslice> resliceSagittalAtSagittal = vtkSmartPointer<vtkImageReslice>::New();
	resliceSagittalAtSagittal->SetInputData(vtkImage);
	resliceSagittalAtSagittal->SetOutputDimensionality(2);


	double sagittalDirectionCosines[9] = { axialX[0], axialX[1], axialX[2],
										axialY[0], axialY[1], axialY[2],
										axialZ[0], axialZ[1], axialZ[2] };

	resliceSagittalAtSagittal->SetResliceAxesDirectionCosines(sagittalDirectionCosines);
	resliceSagittalAtSagittal->SetResliceAxesOrigin(origin_cut);
	resliceSagittalAtSagittal->SetInterpolationModeToLinear();
	resliceSagittalAtSagittal->Update();


	// 更新部分meta信息
	sliceid = 5;
	metafilledpts(sagittalDirectionCosines, origin_vtk, spacing,
		resliceSagittalAtSagittal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImageData* vtkImage_resliceSagittal = resliceSagittalAtSagittal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceSagittal, metaori, dcmIO, value.str(), sliceid);

	// 生成相邻轴位切片，仅改变中心点origin_cut的位置
	// 利用vtk-z计算相邻矢装位切片
	origin_cut[2] = origin_cut[2] - sliceGap;
	resliceSagittalAtSagittal->SetResliceAxesOrigin(origin_cut);
	resliceSagittalAtSagittal->Update();
	// 更新部分meta信息
	sliceid = 4;
	metafilledpts(sagittalDirectionCosines, origin_vtk, spacing,
		resliceSagittalAtSagittal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImage_resliceSagittal = resliceSagittalAtSagittal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceSagittal, metaori, dcmIO, value.str(), sliceid);


	// 利用vtk - z计算相邻矢装位切片
	origin_cut[2] = origin_cut[2] + sliceGap * 2;
	resliceSagittalAtSagittal->SetResliceAxesOrigin(origin_cut);
	resliceSagittalAtSagittal->Update();
	// 更新部分meta信息
	sliceid = 6;
	metafilledpts(sagittalDirectionCosines, origin_vtk, spacing,
		resliceSagittalAtSagittal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImage_resliceSagittal = resliceSagittalAtSagittal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceSagittal, metaori, dcmIO, value.str(), sliceid);
	origin_cut[2] -= sliceGap;



	// 3. MPR得到冠状面
	//axialZ[1] = 0;
	//if (sliceDirection) {
	//	axialZ[0] = 1;
	//}
	//else
	//	axialZ[0] = -1;

	//axialX[0] = 0; axialX[2] = -1;
	//axialY[1] = 0; axialY[2] = 1;
	axialX[0] = 0;
	if (sliceDirection) {
		axialX[2] = -1;
	}
	else
		axialX[2] = 1;
	axialZ[2] = 0; axialZ[0] = 1;
	vtkSmartPointer<vtkImageReslice> resliceCoronalAtCoronal = vtkSmartPointer<vtkImageReslice>::New();
	resliceCoronalAtCoronal->SetInputData(vtkImage);
	resliceCoronalAtCoronal->SetOutputDimensionality(2);

	// 设置切片平面的方向和原点
	double coronalDirectionCosines[9] = { axialX[0], axialX[1], axialX[2],
										axialY[0], axialY[1], axialY[2],
										axialZ[0], axialZ[1], axialZ[2] };

	resliceCoronalAtCoronal->SetResliceAxesDirectionCosines(coronalDirectionCosines);
	resliceCoronalAtCoronal->SetResliceAxesOrigin(origin_cut);
	resliceCoronalAtCoronal->SetInterpolationModeToLinear();
	resliceCoronalAtCoronal->Update();

	// 更新部分特殊meta信息
	sliceid = 8;
	metafilledpts(coronalDirectionCosines, origin_vtk, spacing,
		resliceCoronalAtCoronal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImageData* vtkImage_resliceCoronal = resliceCoronalAtCoronal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceCoronal, metaori, dcmIO, value.str(), sliceid);

	// 生成相邻轴位切片，仅改变中心点origin_cut的位置
	// 利用vtk-x计算生成相邻冠状位切片
	origin_cut[0] = origin_cut[0] - sliceGap;
	resliceCoronalAtCoronal->SetResliceAxesOrigin(origin_cut);
	resliceCoronalAtCoronal->Update();
	// 更新部分特殊meta信息
	sliceid = 7;
	metafilledpts(coronalDirectionCosines, origin_vtk, spacing,
		resliceCoronalAtCoronal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImage_resliceCoronal = resliceCoronalAtCoronal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceCoronal, metaori, dcmIO, value.str(), sliceid);

	// 利用vtk-x计算生成相邻冠状位切片
	origin_cut[0] = origin_cut[0] + sliceGap * 2;
	resliceCoronalAtCoronal->SetResliceAxesOrigin(origin_cut);
	resliceCoronalAtCoronal->Update();
	// 更新部分特殊meta信息
	sliceid = 9;
	metafilledpts(coronalDirectionCosines, origin_vtk, spacing,
		resliceCoronalAtCoronal, metaori, sliceid, sliceDirection);

	// 保存dcm
	vtkImage_resliceCoronal = resliceCoronalAtCoronal->GetOutput();
	value.str(""); value << savedcmPath << (seriesNum + sliceid - 1) << ".dcm";
	SaveAsDCM(vtkImage_resliceCoronal, metaori, dcmIO, value.str(), sliceid);



}




/*
Function: fileMove

Descirption: move dcm file from ori folder to new folder
@param orifolderPath: ori path of the file,
@param newfolderPath: destination path.
*/
void fileMove(string orifolderPath, string newfolderPath) {

	CString last = CString(orifolderPath.data()) + "*.dcm";
	CFileFind tempfind;
	BOOL bFound = tempfind.FindFile(last);
	typedef std::map<int, CString> CFileArray;
	CFileArray HALFilePathArray, FileNameArray, FilePathArray;
	CString filename, filepath;
	int file_num = 0;
	bFound = tempfind.FindFile(last);

	while (bFound) {
		bFound = tempfind.FindNextFile();
		filename = tempfind.GetFileName();
		filepath = tempfind.GetFilePath();
		FileNameArray[file_num] = filename;
		FilePathArray[file_num] = filepath;
		HALFilePathArray[file_num] = CString(newfolderPath.data()) + filename;
		file_num++;

		if (!tempfind.IsDots())
			continue;
	}
	tempfind.Close();
	for (int i = 0; i < file_num; i++) {
		CopyFile(FilePathArray[i].GetBuffer(), HALFilePathArray[i].GetBuffer(), FALSE);
		DeleteFile(FilePathArray[i]);
	}
}




/*
Function: Autolocalizer

Descirption: Use neural network to generate the loclizer dcms 
@param scandataPath: path temp 3d dcms saved;
@param savedcmPath: path final loclizer dcm save in dicom;
@param bodypart: Indicatet the model gonna to be loaded.
				 0 -- brain;
@param fileNum: Indicate the filenum scaned.
*/
extern "C" __declspec(dllexport) int Autolocalizer(string scandataPath, string savedcmPath = "", int bodypart = 0, int fileNum =120) {

	if (fileNum <= 0)
		fileNum = 120;

	itk::MetaDataDictionary metaori;	
	mprOutStruct localize_result;
	Logger logger;
	
	ImageType3F::Pointer itk_image;
	try {
		itk_image = dcmDataLoad(scandataPath, metaori, savedcmPath, fileNum);
		// itk_image = dcmDataLoad_new(scandataPath, metaori);
	}
	catch(...) {
		// worst case, itk wrong
		return -1;
	}
	ImageType3F::Pointer imageData = dataPreprocess(itk_image);

	// Insert noise border detection function
	// noiseDetect(itk_image);

	// use body part flag to decide the model
	string onnx_model_path, engine_model_path;
	if (bodypart == 0) {
		onnx_model_path = "./model.onnx";
		engine_model_path = "./model.engine";
	}
	else {
		// opps, we only got one body part model now...
		std::cout << "Could not match this body part" << std::endl;

		// Insert noise border detection function
		noiseDetect(itk_image);

		mprIsoCenter(itk_image, metaori, savedcmPath + "\\");
		return 4;
	}
	
	int engineBulidflag = 1;
	ICudaEngine* engine = buildEngine(engine_model_path, onnx_model_path, logger, engineBulidflag);
	if (!engineBulidflag) {

		mprIsoCenter(itk_image, metaori, savedcmPath + "\\");

		engine->destroy();
		std::cout << "Error happen engine build part" << std::endl;
		return 3;
	}


	float* output_buffer;// = enginePredict(engine, imageData);	
	try {
		output_buffer = enginePredict(engine, imageData);
	}
	catch (...) {
		//  Any possible thing happened in engine related work
		//  Now we just implement center slice generated method instead...

		// Insert noise border detection function
		noiseDetect(itk_image);
		mprIsoCenter(itk_image, metaori, savedcmPath + "\\");

		engine->destroy();
		std::cout << "Error happen predict part" << std::endl;
		return 2;
	}	

	vector<array<int, 3>> out_points = processOutputBuffer(output_buffer);
	double ofs[9]; double obs[9];

	

	try {
		mprProcess(itk_image, out_points, metaori, obs, ofs, savedcmPath + "\\");
	}
	catch (...) {
		mprIsoCenter(itk_image, metaori, savedcmPath + "\\");

		delete[] output_buffer;
		engine->destroy();
		std::cout << "Error happen in final mpr part" << std::endl;
		return 5;
	}


	// save 9 centerSlice img as backup
	size_t lastSlash = savedcmPath.find_last_of("/\\");
	string pid = savedcmPath.substr(lastSlash + 1);

	//std::ostringstream value;
	std::ostringstream value;
	value.str("");
	for (int i = 0; i < lastSlash; i++)
		value << savedcmPath[i];
	value << "\\Backup\\";
	// string strTempPath = value.str();
	if (_access(value.str().data(), 0) == -1) {
		int ret = _mkdir(value.str().data());
		if (ret < 0) {
			delete[] output_buffer;
			engine->destroy();
			return -3;
		}
	}
	value << pid << "\\";
	if (_access(value.str().data(), 0) == -1) {
		int ret = _mkdir(value.str().data());
		if (ret < 0) {
			delete[] output_buffer;
			engine->destroy();
			return -3;
		}
	}

	mprIsoCenter(itk_image, metaori, value.str());

	// Insert noise border detection function
    noiseDetect(itk_image);
	// value << "1_";
    mprIsoCenter(itk_image, metaori, value.str());
	mprProcess(itk_image, out_points, metaori, obs, ofs, savedcmPath + "\\");

	
	// Temp backup stategy:
	// Incase the model failed to deal with some certain cases:

	std::ostringstream rawDatabackupFolder; rawDatabackupFolder.str("");
	for (int i = 0; i < lastSlash; i++)
		rawDatabackupFolder << savedcmPath[i];
	rawDatabackupFolder << "\\Backup\\RawData\\";

	if (_access(rawDatabackupFolder.str().data(), 0) == -1) {
		int ret = _mkdir(rawDatabackupFolder.str().data());
		if (ret < 0) {
			delete[] output_buffer;
			engine->destroy();
			return -3;
		}
	}
	rawDatabackupFolder << pid << "\\";
	if (_access(rawDatabackupFolder.str().data(), 0) == -1) {
		int ret = _mkdir(rawDatabackupFolder.str().data());
		if (ret < 0) {
			delete[] output_buffer;
			engine->destroy();
			return -3;
		}
	}
	fileMove(scandataPath +"\\", rawDatabackupFolder.str());




	delete[] output_buffer;
	engine->destroy();
	return 1;
}




int autolocalizder_main()
{
	int sidinput = 0;
	std::cout << "Please input the folder sid:"<<endl;
	std::cin >> sidinput;

	//sidinput = 3585;

	std::ostringstream value; value.str("");
	std::ostringstream oridataPath; oridataPath.str("");


	value << ".\\" << sidinput << "\\Temp";
	oridataPath << ".\\" << sidinput << "\\Temp\\";

	string dcm_dir = value.str();//".\\18036\\Temp";
		
	string onnx_model_path = "./model.onnx";
	string engine_model_path = "./model.engine";
	mprOutStruct localize_result;
	int mode = 1;
	Logger logger;
	itk::MetaDataDictionary metaori;

	auto startDataLoad = chrono::high_resolution_clock::now(); // 开始计时
	
	//ImageType3F::Pointer itk_image = dcmDataLoad(dcm_dir, metaori);


	value.str("");
	value << ".\\" << sidinput << "\\Temp";
	string scandataPath = value.str(); //".\\18036\\Temp";

	value.str("");
	value << ".\\" << sidinput;
	string savedcmPath = value.str();//"D:\\SVN\\x64_2023_v1\\Common\\ImgProcess\\auto_brain_location\\auto_brain_location\\18036";
	ImageType3F::Pointer itk_image = dcmDataLoad(scandataPath, metaori, savedcmPath);

	//ImageType3F::Pointer itk_image = dcmDataLoad_new(scandataPath, metaori);
	
	
	// Insert noise border detection function
	// noiseDetect(itk_image);
	


	auto endDataLoad = chrono::high_resolution_clock::now(); // 结束计时
	chrono::duration<double> dataLoadTime = endDataLoad - startDataLoad; // 计算耗时
	cout << "DataLoad Time: " << dataLoadTime.count() << " seconds" << endl;

	auto startPreprocessing = chrono::high_resolution_clock::now(); // 开始计时
	ImageType3F::Pointer imageData = dataPreprocess(itk_image);
	auto endPreprocessing = chrono::high_resolution_clock::now(); // 结束计时
	chrono::duration<double> preProcessingTime = endPreprocessing - startPreprocessing; // 计算耗时
	cout << "Preprocessing Time: " << preProcessingTime.count() << " seconds" << endl;


	float* output_buffer; ICudaEngine* engine;
	// 设置部分临时变量
	//string savedcmPath = ""; 
	double obs[9]; double ofs[9];

	
	auto startBuildEngine = chrono::high_resolution_clock::now(); // 开始计时
	int engineBulidflag; engineBulidflag = 1;
	engine = buildEngine(engine_model_path, onnx_model_path, logger, engineBulidflag);
	auto endBuildEngine = chrono::high_resolution_clock::now(); // 结束计时
	chrono::duration<double> buildEngineTime = endBuildEngine - startBuildEngine; // 计算耗时
	cout << "BuildEngine Time: " << buildEngineTime.count() << " seconds" << endl;
	
	if (!engineBulidflag) {

		mprIsoCenter(itk_image, metaori, savedcmPath + "\\");

		engine->destroy();
		std::cout << "Error happen engine build part" << std::endl;
		return -1;
	}

	
	try {
		auto startEnginePredict = chrono::high_resolution_clock::now(); // 开始计时
		output_buffer = enginePredict(engine, imageData);
		auto endEnginePredict = chrono::high_resolution_clock::now(); // 结束计时
		chrono::duration<double> enginePredictTime = endEnginePredict - startEnginePredict; // 计算耗时
		cout << "EnginePredict Time: " << enginePredictTime.count() << " seconds" << endl;


	}
	catch(...){
		//  Any possible thing happened in engine related work
		//  Now we just implement center slice generated method instead...	

		mprIsoCenter(itk_image, metaori, savedcmPath + "\\");
		//delete[] output_buffer;
		engine->destroy();
		std::cout << "Error happen predict part" << std::endl;
		return -1;
	}


	


	///////////////////////////读取buffer，仅调试用////////////////////////////////
	//ifstream binaryFile("output_buffer_4028.bin", ios::binary | ios::ate);
	//streampos bufferSize = binaryFile.tellg();
	//float* output_buffer = new float[bufferSize / sizeof(float)];
	//binaryFile.seekg(0, std::ios::beg);
	//binaryFile.read(reinterpret_cast<char*>(output_buffer), bufferSize);
	//binaryFile.close();
	///////////////////////////读取buffer，仅调试用////////////////////////////////


	std::cout << "start post process" << std::endl;
	auto startPostProcessing = chrono::high_resolution_clock::now(); // 开始计时
	vector<array<int, 3>> out_points = processOutputBuffer(output_buffer);
	auto endPostProcessing = chrono::high_resolution_clock::now(); // 结束计时
	chrono::duration<double> postProcessingTime = endPostProcessing - startPostProcessing; // 计算耗时
	cout << "PostProcessing Time: " << postProcessingTime.count() << " seconds" << endl;

	auto startMPR = chrono::high_resolution_clock::now(); // 开始计时
	//mprProcess(itk_image, out_points, mode, localize_result);
	std::cout << "start mpr process" << std::endl;
	try {
		mprProcess(itk_image, out_points, metaori, obs, ofs, savedcmPath + "\\");
	}
	catch (...) {
		std::cout << "Error in vtk mpr part" << std::endl;
		int a = 0;
		std::cin >> a;

		mprIsoCenter(itk_image, metaori, savedcmPath + "\\");
		delete[] output_buffer;
		engine->destroy();
		
		return -1;
	}
	auto endMPR = chrono::high_resolution_clock::now(); // 结束计时
	chrono::duration<double> MPRTime = endMPR - startMPR; // 计算耗时
	cout << "MPR Time: " << MPRTime.count() << " seconds" << endl;

	chrono::duration<double> TotalTime = endMPR - startDataLoad; // 计算耗时
	cout << "Total Time: " << TotalTime.count() << " seconds" << endl;

	// save 9 centerSlice img as backup
	//CString strTempPath = savedcmPath.data + "./Backup/";	//建立临时Backup文件夹
	size_t lastSlash = savedcmPath.find_last_of("/\\");
	string pid = savedcmPath.substr(lastSlash + 1);

	//std::ostringstream value;
	value.str("");
	for (int i = 0; i < lastSlash; i++)
		value << savedcmPath[i];
	value << "\\Backup\\";
	// string strTempPath = value.str();
	if (_access(value.str().data(), 0) == -1) {
		int ret = _mkdir(value.str().data());
		if (ret < 0) {
			delete[] output_buffer;
			engine->destroy();
			return -3;
		}
	}
	value << pid << "\\";
	if (_access(value.str().data(), 0) == -1){
		int ret = _mkdir(value.str().data());
		if (ret < 0) {
			delete[] output_buffer;
			engine->destroy();
			return -3;
		}
	}

	mprIsoCenter(itk_image, metaori, value.str());


	// Insert noise border detection function
	noiseDetect(itk_image);
	value << "1_";
	mprIsoCenter(itk_image, metaori, value.str());
	mprProcess(itk_image, out_points, metaori, obs, ofs, savedcmPath + "\\");


	// Temp backup stategy:
	// Incase the model failed to deal with some certain cases:

	std::ostringstream rawDatabackupFolder; rawDatabackupFolder.str("");
	for (int i = 0; i < lastSlash; i++)
		rawDatabackupFolder << savedcmPath[i];
	rawDatabackupFolder << "\\Backup\\RawData\\";

	if (_access(rawDatabackupFolder.str().data(), 0) == -1) {
		int ret = _mkdir(rawDatabackupFolder.str().data());
		if (ret < 0) {
			delete[] output_buffer;
			engine->destroy();
			return -3;
		}
	}
	rawDatabackupFolder << pid << "\\";
	if (_access(rawDatabackupFolder.str().data(), 0) == -1) {
		int ret = _mkdir(rawDatabackupFolder.str().data());
		if (ret < 0) {
			delete[] output_buffer;
			engine->destroy();
			return -3;
		}
	}
	// fileMove(oridataPath.str(), rawDatabackupFolder.str());




	delete[] output_buffer;
	engine->destroy();
	int a = 0;
	std::cin >> a;
	return EXIT_SUCCESS;
}



// noiseDetect main test
int noiseDetect_main() {

	int sidinput = 0;
	std::cout << "Please input the folder sid:" << endl;
	std::cin >> sidinput;

	// sidinput = 3673;

	std::ostringstream value; value.str("");

	value << ".\\" << sidinput << "\\Temp";

	string dcm_dir = value.str();//".\\18036\\Temp";

	
	itk::MetaDataDictionary metaori;


	value.str("");
	value << ".\\" << sidinput << "\\Temp";
	string scandataPath = value.str(); 

	value.str("");
	value << ".\\" << sidinput;
	string savedcmPath = value.str();
	ImageType3F::Pointer itk_image = dcmDataLoad(scandataPath, metaori, savedcmPath);


	// save 9 centerSlice img as backup
	//CString strTempPath = savedcmPath.data + "./Backup/";	//建立临时Backup文件夹
	size_t lastSlash = savedcmPath.find_last_of("/\\");
	string pid = savedcmPath.substr(lastSlash + 1);

	//std::ostringstream value;
	value.str("");
	for (int i = 0; i < lastSlash; i++)
		value << savedcmPath[i];
	value << "\\Backup\\";
	// string strTempPath = value.str();
	if (_access(value.str().data(), 0) == -1) {
		int ret = _mkdir(value.str().data());
		if (ret < 0) 
			return -3;		
	}
	value << pid << "\\";
	if (_access(value.str().data(), 0) == -1) {
		int ret = _mkdir(value.str().data());
		if (ret < 0)
			return -3;
	}

	mprIsoCenter(itk_image, metaori, value.str());


	// Insert noise border detection function
	noiseDetect(itk_image);

	value << "NoiseFiltered_";
	mprIsoCenter(itk_image, metaori, value.str());

	/*int a = 0;
	std::cin >> a;*/
	return EXIT_SUCCESS;
}


/*
Function: Engine_build_main

Descirption: Could generate this exe for Customer Service to rebuild engine in Hospital
	         in case some emergency errors happen.
*/
int Engine_build_main()
{
	string onnx_model_path = "./model.onnx";
	string engine_model_path = "./model.engine";
		
	Logger logger;
	ICudaEngine* engine;

	auto startBuildEngine = chrono::high_resolution_clock::now(); // 开始计时
	int engineBulidflag; engineBulidflag = 1;
	engine = buildEngine(engine_model_path, onnx_model_path, logger, engineBulidflag);
	auto endBuildEngine = chrono::high_resolution_clock::now(); // 结束计时
	chrono::duration<double> buildEngineTime = endBuildEngine - startBuildEngine; // 计算耗时
	cout << "BuildEngine Time: " << buildEngineTime.count() << " seconds" << endl;

	if (!engineBulidflag) {		
		engine->destroy();
		std::cout << "Error happen engine build part" << std::endl;		
		return -1;
	}
	return 1;
}




/*
Function: dcmChange

Descirption: Exchange the certain loclizer dcms to the backup folder 
@param datapath: path of the localizer, e.g.: D:\\APEX\\DATA\\PAT_01\\S_02\\18036;
*/
extern "C" __declspec(dllexport) int dcmChange(string datapath) {

	// first create a temp dir in Backup folder to implement the file exchange 
	size_t lastSlash = datapath.find_last_of("/\\");
	string pid = datapath.substr(lastSlash + 1);

	std::ostringstream tempPath, backupPath;
	tempPath.str(""); backupPath.str("");
	for (int i = 0; i < lastSlash; i++) {
		tempPath << datapath[i];
		backupPath << datapath[i];
	}
	tempPath << "\\Backup\\TempFolder\\";
	backupPath << "\\Backup\\" << pid << "\\";
	
	if (_access(tempPath.str().data(), 0) == -1) {
		int ret = _mkdir(tempPath.str().data());
		if (ret < 0)			
			return -1;		
	}
	if (_access(backupPath.str().data(), 0) == -1)
		return -1;	
	
	// exchange ori 9 dcms & 9 backup dcms
	fileMove(datapath + "\\", tempPath.str());
	fileMove(backupPath.str(),datapath + "\\");
	fileMove(tempPath.str(),backupPath.str());

	return 1;
}

void dcmExchange_main() {

	string datapath = "D:\\SVN\\x64_2023_v1\\Common\\ImgProcess\\auto_brain_location\\auto_brain_location\\18036";
	dcmChange(datapath);
}

/*
int dicomLoad_main() {
	typedef itk::Image<float, 3> ImageType3F;
	itk::GDCMImageIO::Pointer dicomIO = itk::GDCMImageIO::New();

	// 设置DICOM文件夹路径
	std::string dcmPath = "./18036";// "G:/auto_brain_location/auto_brain_location/data/scan2/424";  // 替换为你的DICOM文件夹路径

	// 获取DICOM系列的文件名
	itk::GDCMSeriesFileNames::Pointer namesGenerator = itk::GDCMSeriesFileNames::New();
	namesGenerator->SetUseSeriesDetails(true);
	namesGenerator->AddSeriesRestriction("0008|0021");

	namesGenerator->SetDirectory(dcmPath);

	typedef std::vector<std::string> SeriesIdContainer;
	const SeriesIdContainer & seriesUID = namesGenerator->GetSeriesUIDs();

	// 选择第一个DICOM系列
	std::string seriesIdentifier = seriesUID.begin()->c_str();

	// 获取该系列的所有文件名
	std::vector<std::string> fileNames = namesGenerator->GetFileNames(seriesIdentifier);

	// 创建DICOM图像读取器
	itk::ImageSeriesReader<ImageType3F>::Pointer reader = itk::ImageSeriesReader<ImageType3F>::New();

	itk::MetaDataDictionary metaori;  // 用于存储元数据
	reader->SetFileNames( fileNames );
	reader->SetImageIO(dicomIO);
	reader->Update();
	// 获取元数据

	metaori = dicomIO->GetMetaDataDictionary();

	return 0;

	for (const std::string &fileName : fileNames) {
		try {
			// 为每个文件设置文件名
			reader->SetFileNames({ fileName });
			reader->SetImageIO(dicomIO);
			reader->Update();

			// 将成功读取的图像添加到序列中
			ImageType3F::Pointer image = reader->GetOutput();
			// 在这里添加处理成功读取的图像的代码
			ImageType3F::SizeType origintSize = image->GetLargestPossibleRegion().GetSize();
			orgImgX = origintSize[0];
			orgImgY = origintSize[1];
			orgImgZ = origintSize[2];


			// 获取元数据
			dicomIO->SetFileName(fileName);
			dicomIO->ReadImageInformation();
			metaori = dicomIO->GetMetaDataDictionary();
		}
		catch (itk::ExceptionObject &ex) {
			// 处理异常，例如输出错误消息
			std::cerr << "Exception caught: " << ex << std::endl;
		}
	}



	// 在这里可以继续处理已成功读取的图像或输出它们

	return 0;
}
*/