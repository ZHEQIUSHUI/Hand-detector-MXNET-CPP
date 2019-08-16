#pragma once
#include <iostream>
#include <mxnet-cpp\MxNetCpp.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace mxnet::cpp;
using namespace std;

class HandDetector
{
	Context *Ctx = nullptr;
	Symbol sym_net;
	map<string,NDArray> args;
	map<string, NDArray> aux;

	mxnet::cpp::NDArray data2ndarray(mxnet::cpp::Context ctx, float * data, int batch_size, int num_channels, int height, int width)
	{
		mxnet::cpp::NDArray ret(mxnet::cpp::Shape(batch_size, num_channels, height, width), ctx, false);

		ret.SyncCopyFromCPU(data, batch_size * num_channels * height * width);

		ret.WaitToRead();  //mxnet::cpp::NDArray::WaitAll();

		return ret;
	}

	//select element by indexs in vector
	template <typename T>
	vector<T> selectByindex(vector<T> vec, vector<size_t> idxs)
	{
		vector<T> result;
		for (size_t i = 0; i < idxs.size(); i++)
		{
			result.push_back(vec[idxs[i]]);
		}
		return result;
	}

public:
	HandDetector(bool use_gpu)
	{
		Ctx = use_gpu ? new Context(kGPU, 0) : new Context(kCPU, 0);
	}
	~HandDetector()
	{
		delete Ctx;
	}
	void Loadmodel(string floder, string prefix)
	{
		sym_net = Symbol::Load(floder + "/" + prefix + "-symbol.json");
		std::map<std::string, mxnet::cpp::NDArray> params_age;
		NDArray::Load(floder + "/" + prefix + "-0000.params", nullptr, &params_age);
		for (const auto &k : params_age)
		{
			if (k.first.substr(0, 4) == "aux:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				aux[name] = k.second.Copy(*Ctx);
			}
			if (k.first.substr(0, 4) == "arg:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				args[name] = k.second.Copy(*Ctx);
			}
		}

		// WaitAll is need when we copy data between GPU and the main memory
		mxnet::cpp::NDArray::WaitAll();
	}

	void detect(Mat src,vector<mx_float>& scores,vector<Rect>& bboxes, double scale=1.0, double thresh=0.8)
	{
		Mat normal;
		resize(src, normal, Size(), scale, scale);
		normal.convertTo(normal, CV_32FC3, 1.0 / 255.0);
		normal = (normal - Scalar(0.485, 0.456, 0.406)) / Scalar(0.229, 0.224, 0.225);
		Mat bgr[3];
		split(normal, bgr);
		Mat b_img = bgr[0];
		Mat g_img = bgr[1];
		Mat r_img = bgr[2];
		int len_img = normal.cols*normal.rows;
		float* data_img = new float[normal.channels()*len_img]; //rrrrr...ggggg...bbbbb...
		memcpy(data_img, r_img.data, len_img * sizeof(*data_img));
		memcpy(data_img + len_img, g_img.data, len_img * sizeof(*data_img));
		memcpy(data_img + len_img + len_img, b_img.data, len_img * sizeof(*data_img));

		NDArray data = data2ndarray(*Ctx, data_img, 1, 3, normal.rows, normal.cols);
		args["data"] = data;

		Executor *exec = sym_net.SimpleBind(*Ctx, args, map<string, NDArray>(), map<string, OpReqType>(), aux);
		exec->Forward(false);

		/*vector<mx_float> cls_id;
		exec->outputs[0].SyncCopyToCPU(&cls_id, exec->outputs[0].Size());*/

		vector<mx_float> socres_data;
		exec->outputs[1].SyncCopyToCPU(&socres_data, exec->outputs[1].Size());

		vector<size_t> order;  //which score (> threshold) index
		for (size_t s = 0; s < socres_data.size(); s++)
		{
			if (socres_data[s] > thresh)
			{
				order.push_back(s);
			}
		}

		vector<mx_float> bbox_data;
		vector<Rect> bboxes_t;
		exec->outputs[2].SyncCopyToCPU(&bbox_data, exec->outputs[2].Size());
		for (size_t i = 0; i < bbox_data.size() / 4; i++)
		{
			Rect bbox(
				Point(bbox_data[4 * i] / scale, bbox_data[4 * i + 1] / scale), 
				Point(bbox_data[4 * i + 2] / scale, bbox_data[4 * i + 3] / scale));
			bboxes_t.push_back(bbox);
		}
		bboxes = selectByindex(bboxes_t, order);
		scores = selectByindex(socres_data, order);
	/*	cls_id = selectByindex(cls_id, order);*/
		delete[] data_img;
		delete exec;
	}
private:

};
