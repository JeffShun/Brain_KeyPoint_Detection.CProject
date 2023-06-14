
#include <iostream>
#include <cstdlib>
#include <string>
#include <cassert>
#include <itkImage.h>
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

#define orgImgX 320
#define orgImgY 320
#define orgImgZ 120

#define nPoint 5 
#define imgX 160
#define imgY 160
#define imgZ 64

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;

using ImageType2U = itk::Image<unsigned short, 2>;
using ImageType3F = itk::Image<float, 3>;
using ReaderType = itk::ImageFileReader<ImageType3F>;


struct mprOutStruct {
	array<float, 3> point1;
	array<float, 3> point2;
	array<float, 3> point3;
	array<float, 3> point4;
	array<float, 3> point5;
	vtkSmartPointer<vtkImageData> resliceAxial;
	vtkSmartPointer<vtkImageData> resliceSagittal;
	vtkSmartPointer<vtkImageData> resliceCoronal;
	double tran_angle[9];
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

array<float, 3> ImageToPhysicalPoint(const array<float, 3>& imagePoint, const array<float, 3>& origin, const array<float, 3>& spacing)
{
	array<float, 3> physicalPoint;
	for (int i = 0; i < 3; ++i) {
		physicalPoint[i] = origin[i] + imagePoint[i] * spacing[i];
	}
	return physicalPoint;
}

// 线程函数，计算每个通道的最大值索引
void calculateMaxIndex(const itk::Image<float, 3>::Pointer& channelImage, array<float, 3>& maxIndex)
{
	using ImageSizeType = ImageType3F::SizeType;
	using ImageIndexType = ImageType3F::IndexType;

	float maxValue = numeric_limits<float>::min();

	float scale_x = 1.0 * orgImgX / imgX;
	float scale_y = 1.0 * orgImgY / imgY;
	float scale_z = 1.0 * orgImgZ / imgZ;

	// 遍历图像像素，找到最大值索引
	for (int z = 0; z < imgZ; ++z)
	{
		for (int y = 0; y < imgY; ++y)
		{
			for (int x = 0; x < imgX; ++x)
			{
				ImageIndexType currentIndex;
				currentIndex[0] = x;
				currentIndex[1] = y;
				currentIndex[2] = z;

				float pixelValue = channelImage->GetPixel(currentIndex);
				if (pixelValue > maxValue)
				{
					maxValue = pixelValue;
					maxIndex = { x*scale_x, y*scale_y, z*scale_z };
				}
			}
		}
	}
}

/************************后处理获得关键点坐标*********************************/
vector<array<float, 3>> processOutputBuffer(const float* output_buffer)
{
	using ResampleFilterType = itk::ResampleImageFilter<ImageType3F, ImageType3F>;
	using InterpolatorType = itk::LinearInterpolateImageFunction<ImageType3F, double>;

	// 创建线程列表和最大值索引结果数组
	vector<thread> threads;
	vector<array<float, 3>> maxIndices(nPoint);

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

		// 启动线程，求最大值索引
		threads.emplace_back(calculateMaxIndex, channelImage, ref(maxIndices[channel]));
	}

	// 等待所有线程完成
	for (auto& thread : threads)
	{
		thread.join();
	}
	return maxIndices;
}


void SaveAsNifti(const vtkSmartPointer<vtkImageData>& imageData, const string& outputPath)
{
	// 转换VTK图像为ITK图像
	ImageType2U::Pointer itkImage = ImageType2U::New();

	// 设置ITK图像的大小和像素数据
	ImageType2U::SizeType size;
	size[0] = imageData->GetDimensions()[0];
	size[1] = imageData->GetDimensions()[1];

	itkImage->SetRegions(size);
	itkImage->Allocate();

	for (int y = 0; y < size[1]; ++y)
	{
		for (int x = 0; x < size[0]; ++x)
		{
			itkImage->SetPixel({ x, y }, static_cast<unsigned short>(imageData->GetScalarComponentAsFloat(x, y, 0, 0)));
		}
	}

	// 创建NIfTI图像IO对象
	using ImageIOType = itk::NiftiImageIO;
	ImageIOType::Pointer niftiIO = ImageIOType::New();

	// 创建NIfTI图像写入器
	using WriterType = itk::ImageFileWriter<ImageType2U>;
	WriterType::Pointer writer = WriterType::New();

	// 设置输出路径
	writer->SetFileName(outputPath);

	// 设置图像IO对象
	writer->SetImageIO(niftiIO);

	// 设置图像元数据
	itk::MetaDataDictionary& metaData = writer->GetMetaDataDictionary();
	string patientName = "John Doe";
	string patientID = "12345";
	string studyDescription = "MRI Study";
	string seriesDescription = "Sagittal Plane";
	string sliceThickness = "2.0";

	// 存储元信息到DICOM标签
	itk::EncapsulateMetaData<string>(metaData, "0010|0010", patientName);
	itk::EncapsulateMetaData<string>(metaData, "0010|0020", patientID);
	itk::EncapsulateMetaData<string>(metaData, "0008|1030", studyDescription);
	itk::EncapsulateMetaData<string>(metaData, "0008|103E", seriesDescription);
	itk::EncapsulateMetaData<string>(metaData, "0018|0050", sliceThickness);

	double originX = imageData->GetOrigin()[0];
	double originY = imageData->GetOrigin()[1];
	double spacingX = imageData->GetSpacing()[0];
	double spacingY = imageData->GetSpacing()[1];

	// 设置原点和间距
	ImageType2U::PointType origin;
	origin[0] = originX;
	origin[1] = originY;

	itkImage->SetOrigin(origin);

	ImageType2U::SpacingType spacing;
	spacing[0] = spacingX;
	spacing[1] = spacingY;

	itkImage->SetSpacing(spacing);

	// 执行写入操作
	writer->SetInput(itkImage);
	writer->Update();
}

/************************图像加载*********************************/
ImageType3F::Pointer dataLoad(const string& dcm_dir)
{
	itk::NiftiImageIOFactory::RegisterOneFactory();

	// 创建图像文件读取器
	ReaderType::Pointer reader = ReaderType::New();

	// 设置要读取的文件名 
	reader->SetFileName(dcm_dir);
	reader->Update();
	return reader->GetOutput();
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
	assert(origintSize[0] == orgImgX);
	assert(origintSize[1] == orgImgY);
	assert(origintSize[2] == orgImgZ);

	// 调整图像大小
	typedef itk::ResampleImageFilter<ImageType3F, ImageType3F> ResampleFilterType;
	ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
	resampleFilter->SetInput(rescaler->GetOutput());

	// 定义目标大小
	ImageType3F::SizeType targetSize;
	targetSize[0] = imgX;
	targetSize[1] = imgY;
	targetSize[2] = imgZ;
	resampleFilter->SetSize(targetSize);

	// 计算resize后的spacing
	ImageType3F::SpacingType targetSpacing;
	for (int i = 0; i < 3; ++i) {
		targetSpacing[i] = rescaler->GetOutput()->GetSpacing()[i] * origintSize[i] / targetSize[i];
	}
	resampleFilter->SetOutputSpacing(targetSpacing);

	// 设置输出原点与输入一致
	resampleFilter->SetOutputOrigin(rescaler->GetOutput()->GetOrigin());
	resampleFilter->SetOutputDirection(rescaler->GetOutput()->GetDirection());

	// 使用线性插值
	typedef itk::LinearInterpolateImageFunction<ImageType3F, double> InterpolatorType;
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	resampleFilter->SetInterpolator(interpolator);

	// 更新输出图像
	resampleFilter->Update();

	return resampleFilter->GetOutput();
}

/************************构建TensorRT Engine*********************************/
ICudaEngine* buildEngine(const string& engine_model_path, const string& onnx_model_path, ILogger& logger)
{
	ICudaEngine *engine;
	// 判断是否存在序列化文件
	ifstream engineFile(engine_model_path, ios_base::in | ios::binary);
	if (!engineFile) {
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
	return engine;
}

/************************engine预测*********************************/
float* enginePredict(ICudaEngine *engine, ImageType3F::Pointer imageData)
{
	float* imageData_buffer = imageData->GetBufferPointer();
	// 获取模型输入尺寸并分配GPU内存
	void *buffers[2];
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
	cudaMemcpyAsync(output_buffer, buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	cudaFree(buffers[0]);
	cudaFree(buffers[1]);
	cudaStreamDestroy(stream);
	context->destroy();

	return output_buffer;
}


/************************MPR输出切面*********************************/
//mode: 1:横断面 2:矢状面  3:冠状面
void mprProcess(ImageType3F::Pointer itk_image, vector<array<float, 3>> out_points, int mode, mprOutStruct& localize_result)
{
	// itk转化为vtk数据
	using ITKToVTKFilterType = itk::ImageToVTKImageFilter<ImageType3F>;
	ITKToVTKFilterType::Pointer itkToVtkFilter = ITKToVTKFilterType::New();
	itkToVtkFilter->SetInput(itk_image);
	itkToVtkFilter->Update();
	vtkImageData* vtkImage = itkToVtkFilter->GetOutput();

	array<float, 3> origin = {
		static_cast<float>(vtkImage->GetOrigin()[0]),
		static_cast<float>(vtkImage->GetOrigin()[1]),
		static_cast<float>(vtkImage->GetOrigin()[2])
	};
	array<float, 3> spacing = {
		static_cast<float>(vtkImage->GetSpacing()[0]),
		static_cast<float>(vtkImage->GetSpacing()[1]),
		static_cast<float>(vtkImage->GetSpacing()[2])
	};

	array<float, 3> point1 = ImageToPhysicalPoint(out_points[0], origin, spacing);
	array<float, 3> point2 = ImageToPhysicalPoint(out_points[1], origin, spacing);
	array<float, 3> point3 = ImageToPhysicalPoint(out_points[2], origin, spacing);
	array<float, 3> point4 = ImageToPhysicalPoint(out_points[3], origin, spacing);
	array<float, 3> point5 = ImageToPhysicalPoint(out_points[4], origin, spacing);

	localize_result.point1 = point1;
	localize_result.point2 = point2;
	localize_result.point3 = point3;
	localize_result.point4 = point4;
	localize_result.point5 = point5;

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

	// 如果是定位横断面，则以横断面为标准，其它两个面与之垂直
	if (mode == 1)
	{
		array<float, 3> axis1_axial = v1_axial;
		array<float, 3> axis2_axial = cal_norm_vec(v1_axial, v2_axial);
		array<float, 3> axis3_axial = v2_axial;
		double origin[3] = { point5[0], point5[1], point5[2] };

		// MPR得到横断面
		vtkSmartPointer<vtkImageReslice> resliceAxialAtAxial = vtkSmartPointer<vtkImageReslice>::New();
		resliceAxialAtAxial->SetInputData(vtkImage);
		resliceAxialAtAxial->SetOutputDimensionality(2);

		// 设置切片平面的方向和原点
		double axialDirectionCosines[9] = { axis3_axial[0], axis3_axial[1], axis3_axial[2],
											-axis1_axial[0], -axis1_axial[1], -axis1_axial[2],
											axis2_axial[0], axis2_axial[1], axis2_axial[2] };

		resliceAxialAtAxial->SetResliceAxesDirectionCosines(axialDirectionCosines);
		resliceAxialAtAxial->SetResliceAxesOrigin(origin);
		resliceAxialAtAxial->SetInterpolationModeToLinear();
		resliceAxialAtAxial->Update();

		// MPR得到矢状面
		vtkSmartPointer<vtkImageReslice> resliceSagittalAtAxial = vtkSmartPointer<vtkImageReslice>::New();
		resliceSagittalAtAxial->SetInputData(vtkImage);
		resliceSagittalAtAxial->SetOutputDimensionality(2);

		// 设置切片平面的方向和原点
		double sagittalDirectionCosines[9] = { axis1_axial[0], axis1_axial[1], axis1_axial[2],
											   axis2_axial[0], axis2_axial[1], axis2_axial[2],
											   -axis3_axial[0], -axis3_axial[1], -axis3_axial[2] };

		resliceSagittalAtAxial->SetResliceAxesDirectionCosines(sagittalDirectionCosines);
		resliceSagittalAtAxial->SetResliceAxesOrigin(origin);
		resliceSagittalAtAxial->SetInterpolationModeToLinear();
		resliceSagittalAtAxial->Update();

		// MPR得到冠状面
		vtkSmartPointer<vtkImageReslice> resliceCoronalAtAxial = vtkSmartPointer<vtkImageReslice>::New();
		resliceCoronalAtAxial->SetInputData(vtkImage);
		resliceCoronalAtAxial->SetOutputDimensionality(2);

		// 设置切片平面的方向和原点
		double coronalDirectionCosines[9] = { axis3_axial[0], axis3_axial[1], axis3_axial[2],
											  axis2_axial[0], axis2_axial[1], axis2_axial[2],
											  axis1_axial[0], axis1_axial[1], axis1_axial[2] };

		resliceCoronalAtAxial->SetResliceAxesDirectionCosines(coronalDirectionCosines);
		resliceCoronalAtAxial->SetResliceAxesOrigin(origin);
		resliceCoronalAtAxial->SetInterpolationModeToLinear();
		resliceCoronalAtAxial->Update();

		vtkImageData* vtkImage_resliceAxial = resliceAxialAtAxial->GetOutput();
		vtkImageData* vtkImage_resliceSagittal = resliceSagittalAtAxial->GetOutput();
		vtkImageData* vtkImage_resliceCoronal = resliceCoronalAtAxial->GetOutput();

		localize_result.resliceAxial = vtkImage_resliceAxial;
		localize_result.resliceSagittal = vtkImage_resliceSagittal;
		localize_result.resliceCoronal = vtkImage_resliceCoronal;

		// 保存切面图像
		string Filename_resliceAxialAtAxial = "E:/ShenFile/Code/brain_oritation_detection/cpp/result/9103/resliceAxialAtAxial.nii.gz";
		string Filename_resliceSagittalAtAxial = "E:/ShenFile/Code/brain_oritation_detection/cpp/result/9103/resliceSagittalAtAxial.nii.gz";
		string Filename_resliceCoronalAtAxial = "E:/ShenFile/Code/brain_oritation_detection/cpp/result/9103/resliceCoronalAtAxial.nii.gz";
		SaveAsNifti(vtkImage_resliceAxial, Filename_resliceAxialAtAxial);
		SaveAsNifti(vtkImage_resliceSagittal, Filename_resliceSagittalAtAxial);
		SaveAsNifti(vtkImage_resliceCoronal, Filename_resliceCoronalAtAxial);
	}

	// 如果是定位矢状面，则以矢状面为标准，其它两个面与之垂直
	else if (mode == 2)
	{
		array<float, 3> axis1_sagittal = v1_sagittal;
		array<float, 3> axis2_sagittal = cal_norm_vec(axis1_sagittal, normal_sagittal);
		array<float, 3> axis3_sagittal = normal_sagittal;
		double origin[3] = { point5[0], point5[1], point5[2] };

		// MPR得到横断面
		vtkSmartPointer<vtkImageReslice> resliceAxialAtSagittal = vtkSmartPointer<vtkImageReslice>::New();
		resliceAxialAtSagittal->SetInputData(vtkImage);
		resliceAxialAtSagittal->SetOutputDimensionality(2);

		// 设置切片平面的方向和原点
		double axialDirectionCosines[9] = { axis3_sagittal[0], axis3_sagittal[1], axis3_sagittal[2],
										    axis1_sagittal[0], axis1_sagittal[1], axis1_sagittal[2],
										    axis2_sagittal[0], axis2_sagittal[1], axis2_sagittal[2] };

		resliceAxialAtSagittal->SetResliceAxesDirectionCosines(axialDirectionCosines);
		resliceAxialAtSagittal->SetResliceAxesOrigin(origin);
		resliceAxialAtSagittal->SetInterpolationModeToLinear();
		resliceAxialAtSagittal->Update();

		// MPR得到矢状面
		vtkSmartPointer<vtkImageReslice> resliceSagittalAtSagittal = vtkSmartPointer<vtkImageReslice>::New();
		resliceSagittalAtSagittal->SetInputData(vtkImage);
		resliceSagittalAtSagittal->SetOutputDimensionality(2);

		// 设置切片平面的方向和原点
		double sagittalDirectionCosines[9] = { axis1_sagittal[0], axis1_sagittal[1], axis1_sagittal[2],
											   axis2_sagittal[0], axis2_sagittal[1], axis2_sagittal[2],
											   -axis3_sagittal[0], -axis3_sagittal[1], -axis3_sagittal[2] };

		resliceSagittalAtSagittal->SetResliceAxesDirectionCosines(sagittalDirectionCosines);
		resliceSagittalAtSagittal->SetResliceAxesOrigin(origin);
		resliceSagittalAtSagittal->SetInterpolationModeToLinear();
		resliceSagittalAtSagittal->Update();

		// MPR得到冠状面
		vtkSmartPointer<vtkImageReslice> resliceCoronalAtSagittal = vtkSmartPointer<vtkImageReslice>::New();
		resliceCoronalAtSagittal->SetInputData(vtkImage);
		resliceCoronalAtSagittal->SetOutputDimensionality(2);

		// 设置切片平面的方向和原点
		double coronalDirectionCosines[9] = { axis3_sagittal[0], axis3_sagittal[1], axis3_sagittal[2],
											  axis2_sagittal[0], axis2_sagittal[1], axis2_sagittal[2],
											  axis1_sagittal[0], axis1_sagittal[1], axis1_sagittal[2] };

		resliceCoronalAtSagittal->SetResliceAxesDirectionCosines(coronalDirectionCosines);
		resliceCoronalAtSagittal->SetResliceAxesOrigin(origin);
		resliceCoronalAtSagittal->SetInterpolationModeToLinear();
		resliceCoronalAtSagittal->Update();

		vtkImageData* vtkImage_resliceAxial = resliceAxialAtSagittal->GetOutput();
		vtkImageData* vtkImage_resliceSagittal = resliceSagittalAtSagittal->GetOutput();
		vtkImageData* vtkImage_resliceCoronal = resliceCoronalAtSagittal->GetOutput();

		localize_result.resliceAxial = vtkImage_resliceAxial;
		localize_result.resliceSagittal = vtkImage_resliceSagittal;
		localize_result.resliceCoronal = vtkImage_resliceCoronal;

		// 保存切面图像
		string Filename_resliceAxialAtSagittal = "E:/ShenFile/Code/brain_oritation_detection/cpp/result/9103/resliceAxialAtSagittal.nii.gz";
		string Filename_resliceSagittalAtSagittal = "E:/ShenFile/Code/brain_oritation_detection/cpp/result/9103/resliceSagittalAtSagittal.nii.gz";
		string Filename_resliceCoronalAtSagittal = "E:/ShenFile/Code/brain_oritation_detection/cpp/result/9103/resliceCoronalAtSagittal.nii.gz";
		SaveAsNifti(vtkImage_resliceAxial, Filename_resliceAxialAtSagittal);
		SaveAsNifti(vtkImage_resliceSagittal, Filename_resliceSagittalAtSagittal);
		SaveAsNifti(vtkImage_resliceCoronal, Filename_resliceCoronalAtSagittal);
	}

	// 如果是定冠状面，则以冠状面为标准，其它两个面与之垂直
	else if (mode == 3)
	{
		array<float, 3> axis1_coronal = cal_norm_vec(v2_coronal, v1_coronal);
		array<float, 3> axis2_coronal = v1_coronal;
		array<float, 3> axis3_coronal = v2_coronal;
		double origin[3] = { point2[0], point2[1], point2[2] };

		// MPR得到横断面
		vtkSmartPointer<vtkImageReslice> resliceAxialAtCoronal = vtkSmartPointer<vtkImageReslice>::New();
		resliceAxialAtCoronal->SetInputData(vtkImage);
		resliceAxialAtCoronal->SetOutputDimensionality(2);

		// 设置切片平面的方向和原点
		double axialDirectionCosines[9] = { axis3_coronal[0], axis3_coronal[1], axis3_coronal[2],
											-axis1_coronal[0], -axis1_coronal[1], -axis1_coronal[2],
											axis2_coronal[0], axis2_coronal[1], axis2_coronal[2] };

		resliceAxialAtCoronal->SetResliceAxesDirectionCosines(axialDirectionCosines);
		resliceAxialAtCoronal->SetResliceAxesOrigin(origin);
		resliceAxialAtCoronal->SetInterpolationModeToLinear();
		resliceAxialAtCoronal->Update();

		// MPR得到矢状面
		vtkSmartPointer<vtkImageReslice> resliceSagittalAtCoronal = vtkSmartPointer<vtkImageReslice>::New();
		resliceSagittalAtCoronal->SetInputData(vtkImage);
		resliceSagittalAtCoronal->SetOutputDimensionality(2);

		// 设置切片平面的方向和原点
		double sagittalDirectionCosines[9] = { axis1_coronal[0], axis1_coronal[1], axis1_coronal[2],
											   axis2_coronal[0], axis2_coronal[1], axis2_coronal[2],
											   -axis3_coronal[0], -axis3_coronal[1], -axis3_coronal[2] };

		resliceSagittalAtCoronal->SetResliceAxesDirectionCosines(sagittalDirectionCosines);
		resliceSagittalAtCoronal->SetResliceAxesOrigin(origin);
		resliceSagittalAtCoronal->SetInterpolationModeToLinear();
		resliceSagittalAtCoronal->Update();

		// MPR得到冠状面
		vtkSmartPointer<vtkImageReslice> resliceCoronalAtCoronal = vtkSmartPointer<vtkImageReslice>::New();
		resliceCoronalAtCoronal->SetInputData(vtkImage);
		resliceCoronalAtCoronal->SetOutputDimensionality(2);

		// 设置切片平面的方向和原点
		double coronalDirectionCosines[9] = { axis3_coronal[0], axis3_coronal[1], axis3_coronal[2],
											  axis2_coronal[0], axis2_coronal[1], axis2_coronal[2],
											  axis1_coronal[0], axis1_coronal[1], axis1_coronal[2] };

		resliceCoronalAtCoronal->SetResliceAxesDirectionCosines(coronalDirectionCosines);
		resliceCoronalAtCoronal->SetResliceAxesOrigin(origin);
		resliceCoronalAtCoronal->SetInterpolationModeToLinear();
		resliceCoronalAtCoronal->Update();

		vtkImageData* vtkImage_resliceAxial = resliceAxialAtCoronal->GetOutput();
		vtkImageData* vtkImage_resliceSagittal = resliceSagittalAtCoronal->GetOutput();
		vtkImageData* vtkImage_resliceCoronal = resliceCoronalAtCoronal->GetOutput();

		localize_result.resliceAxial = vtkImage_resliceAxial;
		localize_result.resliceSagittal = vtkImage_resliceSagittal;
		localize_result.resliceCoronal = vtkImage_resliceCoronal;

		// 保存切面图像
		string Filename_resliceAxialAtCoronal = "E:/ShenFile/Code/brain_oritation_detection/cpp/result/9103/resliceAxialAtCoronal.nii.gz";
		string Filename_resliceSagittalAtCoronal = "E:/ShenFile/Code/brain_oritation_detection/cpp/result/9103/resliceSagittalAtCoronal.nii.gz";
		string Filename_resliceCoronalAtCoronal = "E:/ShenFile/Code/brain_oritation_detection/cpp/result/9103/resliceCoronalAtCoronal.nii.gz";
		SaveAsNifti(vtkImage_resliceAxial, Filename_resliceAxialAtCoronal);
		SaveAsNifti(vtkImage_resliceSagittal, Filename_resliceSagittalAtCoronal);
		SaveAsNifti(vtkImage_resliceCoronal, Filename_resliceCoronalAtCoronal);
	}
}

int main()
{
	string dcm_dir = "E:/ShenFile/Code/brain_oritation_detection/test_data/dcm/9103.nii.gz";
	string onnx_model_path = "E:/ShenFile/Code/brain_oritation_detection/Brain_KeyPoint_Detection/train/checkpoints/onnx_model/model.onnx";
	string engine_model_path = "./model.engine";
	mprOutStruct localize_result;
	int mode = 3;
	Logger logger;

	ImageType3F::Pointer itk_image = dataLoad(dcm_dir);
	ImageType3F::Pointer imageData = dataPreprocess(itk_image);
	ICudaEngine* engine = buildEngine(engine_model_path, onnx_model_path, logger);
	float* output_buffer = enginePredict(engine, imageData);
	vector<array<float, 3>> out_points = processOutputBuffer(output_buffer);
	mprProcess(itk_image, out_points, mode, localize_result);

	delete[] output_buffer;
	engine->destroy();

	return EXIT_SUCCESS;
}

