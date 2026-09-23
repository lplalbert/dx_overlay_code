"""
本代码主要实现生成双通道（U和V通道）水印模板的功能。
主要功能包括：
1. 定义了生成基本水印单元（高斯圆型和矩形）的函数。
2. 实现了单块水印的生成逻辑，分别控制U和V通道的嵌入强度。
3. 提供了基于Reed-Solomon编码将数字转换为水印序列的功能。
4. 实现了WM_Template_Generator类，用于根据预设矩阵生成16种基础水印模板。
5. 包含生成完整屏幕水印图像的逻辑，将水印信息块按照行列布局组合成最终的图像。
6. 支持生成正向和反向两种水印模板。
"""
import numpy as np
import cv2
import os
import argparse
import reedsolo
import functools


WATERMARK_GRID_SIZE = 8
WATERMARK_GRID_CELLS = WATERMARK_GRID_SIZE * WATERMARK_GRID_SIZE
# Every pixel carries this alpha marker so the C++ loader can distinguish the
# lossless channel-data format from legacy RGB/RGBA display images.
CHANNEL_ENCODING_ALPHA = 254


########## 生成单个水印bit 中的单元的形态 ##########
@functools.lru_cache(maxsize=None)
def gen_gaussian_tp(blockRow):
	"""
	高斯圆型，生成一个中间圆形区域是固定的像素是，周围是渐变的水印的形态。
	"""
	template = np.zeros((blockRow, blockRow), dtype=np.float32)
	CenterX = blockRow // 2

	y, x = np.ogrid[:blockRow, :blockRow]
	dist = np.sqrt((y - CenterX)**2 + (x - CenterX)**2)
	Radius = 1.0 - dist / CenterX

	mask1 = (Radius > 0) & (Radius <= 0.3)
	mask2 = Radius > 0.3

	template[mask1] = np.round(125 * np.sqrt(Radius[mask1]) * 1.25)
	template[mask2] = 125
	return template.astype(np.uint8)

@functools.lru_cache(maxsize=None)
def gen_rect_tp(blockRow):
	"""
	恢复到原始整单元矩形矩阵。
	矩形本身覆盖完整单元。
	"""
	template = np.ones((blockRow, blockRow), dtype=np.uint8) * 125
	return template

@functools.lru_cache(maxsize=None)
def gen_gaussian_tp_v2(blockRow, flat_ratio=0.7):
	"""
	宽平高斯型，增加中间满信号区域的面积。
	flat_ratio: 中间满信号区域占半径的比例
	"""
	template = np.zeros((blockRow, blockRow), dtype=np.float32)
	CenterX = blockRow // 2
	y, x = np.ogrid[:blockRow, :blockRow]
	dist = np.sqrt((y - CenterX)**2 + (x - CenterX)**2)
	Radius = 1.0 - dist / CenterX

	mask_flat = Radius > (1.0 - flat_ratio)
	mask_grad = (Radius > 0) & (~mask_flat)

	template[mask_flat] = 125
	# 在梯度区域进行平滑过渡
	if (1.0 - flat_ratio) > 0:
		norm_grad = Radius[mask_grad] / (1.0 - flat_ratio)
		template[mask_grad] = np.round(125 * np.sqrt(norm_grad) * 1.25)

	template = np.clip(template, 0, 125)
	return template.astype(np.uint8)

@functools.lru_cache(maxsize=None)
def gen_soft_rect_tp(blockRow, border=2):
	"""
	软边矩形，减少周边高频跳变。
	"""
	template = np.ones((blockRow, blockRow), dtype=np.float32) * 125
	mask = np.ones((blockRow, blockRow), dtype=np.float32)
	for i in range(border):
		val = (i + 1) / (border + 1)
		mask[i, :] *= val
		mask[-1-i, :] *= val
		mask[:, i] *= val
		mask[:, -1-i] *= val
	return (template * mask).astype(np.uint8)



################# 生成单个水印比特的块， 水印0 和1 的水印块#####################

def gen_block_single_uv_decouple(k = 1, kv = 1, blockRow = 32,
						   ratio_u = 10,
						   ratio_v = 8,
						   u_tp_fn = gen_rect_tp,
						   v_tp_fn = gen_gaussian_tp
						   ):
	"""
	返回 BGRA 数据纹理，PNG 解码后的逻辑通道为 R=Y、G=Cr、B=Cb、A=254。
	OpenCV 的 YCrCb 顺序是 Y/Cr/Cb；旧 ratio_u 实际控制动态 Cr，
	旧 ratio_v 实际控制静态 Cb。保留参数名仅用于兼容旧调用。
	"""
	y = np.full((blockRow, blockRow), 128, dtype=np.int16)
	cr = np.full((blockRow, blockRow), 128, dtype=np.int16)
	cb = np.full((blockRow, blockRow), 128, dtype=np.int16)

	####### dynamic Cr channel (legacy ratio_u) #########
	if ratio_u != 0:
		cr_delta = u_tp_fn(blockRow).astype(np.int16) * ratio_u // 10
		if k == 1:
			cr += cr_delta
		else:
			cr -= cr_delta

	########## static Cb channel (legacy ratio_v) #########
	if ratio_v != 0:
		cb_delta = v_tp_fn(blockRow).astype(np.int16) * ratio_v // 10
		if kv == 1:
			cb += cb_delta
		else:
			cb -= cb_delta

	y = np.clip(y, 0, 255).astype(np.uint8)
	cr = np.clip(cr, 0, 255).astype(np.uint8)
	cb = np.clip(cb, 0, 255).astype(np.uint8)
	alpha = np.full((blockRow, blockRow), CHANNEL_ENCODING_ALPHA, dtype=np.uint8)
	# cv2.imwrite accepts BGRA. This ordering produces logical PNG RGBA=Y/Cr/Cb/marker.
	return cv2.merge((cb, cr, y, alpha))


def gen_wm_blocks_uv( message_seq, type = 1 , gen_fn = gen_block_single_uv_decouple,
				  single_block_size = 64,
				  ratio_u =10,
				  ratio_v =8,
				  inverse = False,
				  u_tp_fn = gen_rect_tp,
				  v_tp_fn = gen_gaussian_tp
				  ):
	seq = np.asarray(message_seq).reshape(-1)
	if seq.size != WATERMARK_GRID_CELLS:
		raise ValueError(
			f"8x8 watermark templates require exactly {WATERMARK_GRID_CELLS} cells, "
			f"got {seq.size}"
		)

	if type == 0:
		if not inverse:
			template_1 = gen_fn(k = 1, kv = 1 , ratio_u=ratio_u, ratio_v=ratio_v, blockRow= single_block_size, u_tp_fn=u_tp_fn, v_tp_fn=v_tp_fn)
			template_0 = gen_fn(k = 0, kv = 0 , ratio_u=ratio_u, ratio_v=ratio_v, blockRow= single_block_size, u_tp_fn=u_tp_fn, v_tp_fn=v_tp_fn)
		else:
			template_1 = gen_fn(k = 0, kv = 1, ratio_u=ratio_u, ratio_v=ratio_v, blockRow= single_block_size, u_tp_fn=u_tp_fn, v_tp_fn=v_tp_fn)
			template_0 = gen_fn(k = 1, kv = 0, ratio_u=ratio_u, ratio_v=ratio_v, blockRow= single_block_size, u_tp_fn=u_tp_fn, v_tp_fn=v_tp_fn)
	else:
		if not inverse:
			template_1 = gen_fn(k = 1, kv = 0 , ratio_u=ratio_u, ratio_v=ratio_v, blockRow= single_block_size, u_tp_fn=u_tp_fn, v_tp_fn=v_tp_fn)
			template_0 = gen_fn(k = 0, kv = 1, ratio_u=ratio_u, ratio_v=ratio_v, blockRow= single_block_size, u_tp_fn=u_tp_fn, v_tp_fn=v_tp_fn)
		else:
			template_1 = gen_fn(k = 0, kv = 0, ratio_u=ratio_u, ratio_v=ratio_v, blockRow= single_block_size, u_tp_fn=u_tp_fn, v_tp_fn=v_tp_fn)
			template_0 = gen_fn(k = 1, kv = 1, ratio_u=ratio_u, ratio_v=ratio_v, blockRow= single_block_size, u_tp_fn=u_tp_fn, v_tp_fn=v_tp_fn)

	img_blocks = np.empty((WATERMARK_GRID_SIZE * single_block_size,
						WATERMARK_GRID_SIZE * single_block_size, 4), dtype = np.uint8)

	for j , bit in enumerate(seq):
		row_idx, col_idx = divmod(j, WATERMARK_GRID_SIZE)
		r0, r1 = row_idx * single_block_size, (row_idx + 1) * single_block_size
		c0, c1 = col_idx * single_block_size, (col_idx + 1) * single_block_size
		# 处理 fix_fg_matrix 中的 -1/1
		bit_val = 1 if bit == 1 else 0
		img_blocks[r0:r1, c0:c1, :] = template_1 if bit_val == 1 else template_0

	return img_blocks

########### 根据提供的 FixFgLib 中的矩阵，生成对应的水印的模板的内容 ###########

class WM_Template_Generator:
	def __init__(self,
			  ratio_u=10,
			  ratio_v=8,
			  type =1,
			  v_tp_fn = gen_gaussian_tp
			  	):
		self.fix_fg_matrix = np.array([
			[1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
			[-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
			[-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
			[-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
			[-1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1],
			[-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
			[1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
			[-1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
			[1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
			[1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
			[-1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
			[-1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
			[1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1],
			[1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
			[1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
			[-1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1]
		])
		self.ratio_u = ratio_u
		self.ratio_v = ratio_v
		self.type = type # control the v channel type
		self.v_tp_fn = v_tp_fn
		################## 初始化16个水印的模板  ################
		self.wm_template_imgs = [None] * 16
		self.wm_template_imgs_inverse = [None] * 16

		self.__gen_16_wm_templates__(inverse = False)
		self.__gen_16_wm_templates__(inverse = True)



	def __gen_16_wm_templates__(self, inverse = False):
		for i in range(16):
			wm_templte = gen_wm_blocks_uv(self.fix_fg_matrix[i],
										single_block_size=64,
										ratio_u=self.ratio_u,
										ratio_v=self.ratio_v,
										type = self.type,
										u_tp_fn = gen_rect_tp,
										v_tp_fn = self.v_tp_fn,
										inverse = inverse)
			if not inverse:
				self.wm_template_imgs[i] = wm_templte
			else:
				self.wm_template_imgs_inverse[i] = wm_templte


	def __call__(self, nums, inverse = False ):
		if not inverse:
			return self.wm_template_imgs[nums]
		else:
			return self.wm_template_imgs_inverse[nums]


################# From RsCode_decoder.py #################

# 模块级别初始化 RS 编解码器（仅初始化一次）
# RS(15,5) over GF(2^4), 与原 galois 库参数完全一致: fcr=1, prim=0x13(x^4+x+1), generator=2
_rs_codec = reedsolo.RSCodec(nsym=10, nsize=15, c_exp=4, fcr=1, prim=0x13, generator=2)
WATERMARK_PAYLOAD_HEX_DIGITS = 5
MAX_WATERMARK_ID = 16 ** WATERMARK_PAYLOAD_HEX_DIGITS - 1

def encode(data):
	# RS(15,5) 编码, 使用 reedsolo 替代 galois
	# nsym=10 表示 10 个校验符号, nsize=15 码字长度, c_exp=4 即 GF(2^4)
	encode_data = list(_rs_codec.encode(data))
	return encode_data

def nums2_16(nums):
	"""
	将给定的任意数字转换为对应的16进制
	"""
	if nums < 0:
		raise ValueError("watermark_id must be non-negative")
	if nums > MAX_WATERMARK_ID:
		raise ValueError(
			f"watermark_id={nums} exceeds the current RS(15,5) payload range "
			f"0..{MAX_WATERMARK_ID} (0x{MAX_WATERMARK_ID:05X}); "
			"use a smaller ID or redesign the encoder/decoder payload size"
		)
	ans = [0]* 5

	i = 0
	while nums > 0:
		ans[i] = nums % 16
		# ans.append(nums % 16 )
		nums = nums //16
		i += 1
	return ans

def get_wm_seq(nums):
	data = nums2_16(nums)
	rscode_data = encode(data[::-1])
	rscode_data = [int(e) for e in rscode_data ]
	# watermark embedding seq
	wm_seq = rscode_data[::-1] + [0]
	return wm_seq

################# From gen_wm_message_dual.py #################

MESSAGE_ROW, MESSAGE_COL = 2, 2

def resize_template_image(image, width, height):
	"""Resize channel-data templates without converting their encoded channels."""
	return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def new_channel_canvas(height, width, channels):
	if channels != 4:
		raise ValueError("channel-data templates must use four BGRA channels")
	canvas = np.empty((height, width, channels), dtype=np.uint8)
	canvas[:, :, :3] = 128
	canvas[:, :, 3] = CHANNEL_ENCODING_ALPHA
	return canvas


def gen_message_block(messages, wm_images, wm_images_inverse, block_height, block_width, message_height, message_width):
	channels = wm_images[0].shape[2]
	message_block = new_channel_canvas(block_height, block_width, channels)
	message_block_inverse = new_channel_canvas(block_height, block_width, channels)
	a, b, c, d = messages[0], messages[1], messages[2], messages[3]
	messages = [a, c, b, d]
	for i in range(MESSAGE_ROW):
		for j in range(MESSAGE_COL):
			index = i * MESSAGE_COL + j
			index = index % len(messages)
			img_type = messages[index]
			wm_image = resize_template_image(wm_images[img_type], message_width, message_height)
			message_block[
				i * message_height : (i + 1) * message_height,
				j * message_width : (j + 1) * message_width,
			] = wm_image

			wm_image_inverse = resize_template_image(
				wm_images_inverse[img_type], message_width, message_height
			)
			message_block_inverse[
				i * message_height : (i + 1) * message_height,
				j * message_width : (j + 1) * message_width,
			] = wm_image_inverse
	return message_block, message_block_inverse


def gen_watermark_imgs(all_messages, block_rows, block_cols, block_height, block_width, message_height, message_width, wm_images, wm_images_inverse):
	channels = wm_images[0].shape[2]
	wm_blocks = [None] * 4
	wm_blocks_inverse = [None] * 4

	for i in range(len(all_messages)):
		wm_msg_block, wm_msg_block_inverse = gen_message_block(
			all_messages[i],
			wm_images,
			wm_images_inverse,
			block_height,
			block_width,
			message_height,
			message_width,
		)
		wm_blocks[i] = wm_msg_block
		wm_blocks_inverse[i] = wm_msg_block_inverse

	wm_blank = new_channel_canvas(block_rows * block_height, block_width * block_cols, channels)
	wm_blank_inverse = new_channel_canvas(block_rows * block_height, block_width * block_cols, channels)

	for i in range(block_rows):
		for j in range(block_cols):
			k = j
			if i % 2:
				k += 2
			k = k % 4
			wm_blank[
				i * block_height : (i + 1) * block_height,
				j * block_width : (j + 1) * block_width,
			] = wm_blocks[k]
			wm_blank_inverse[
				i * block_height : (i + 1) * block_height,
				j * block_width : (j + 1) * block_width,
			] = wm_blocks_inverse[k]
	return wm_blank, wm_blank_inverse


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--nums", type=int, default=123456, help="the number to encode")
	parser.add_argument("--block_rows", '-row', type=int, default=4, help="block rows")
	parser.add_argument("--block_cols", '-col', type=int, default=6, help="block cols")
	parser.add_argument("--screen_width", type=int, default=1920, help="screen width")
	parser.add_argument("--screen_height", type=int, default=1080, help="screen height")
	parser.add_argument("--save_dir", type=str, default="generated_templates", help="directory to save templates")
	parser.add_argument("--pattern", type=str, default="gaussian_v2", choices=["gaussian", "gaussian_v2", "soft_rect", "rect"], help="static Cb sub-block pattern; dynamic Cr stays rectangular")
	parser.add_argument("--dynamic_ratio_cr", "--ratio_u", dest="ratio_u", type=int, default=10, help="dynamic Cr generation amplitude (legacy: ratio_u)")
	parser.add_argument("--static_ratio_cb", "--ratio_v", dest="ratio_v", type=int, default=8, help="static Cb generation amplitude (legacy: ratio_v)")
	parser.add_argument("--train", type=int, default=None, choices=range(0, 16), help="训练模式：输入0-15的数字，跳过RS编码，所有位置填充同一码字")

	parser.add_argument("--type_val", type=int, default=0, choices=[0, 1], help="watermark pattern type")
	args = parser.parse_args()
	# Mapping patterns to generation functions
	pattern_map = {
		"gaussian": gen_gaussian_tp,
		"gaussian_v2": gen_gaussian_tp_v2,
		"soft_rect": gen_soft_rect_tp,
		"rect": gen_rect_tp,
	}
	chosen_pattern_fn = pattern_map[args.pattern]
	# 1. Get watermark sequence from number
	if args.train is not None:
		# 训练模式：跳过RS编码，所有位置填充同一码字
		wm_seq = [args.train] * 16
		print(f"Train mode: all codewords set to {args.train}, sequence: {wm_seq}")
	else:
		wm_seq = get_wm_seq(args.nums)
		print(f"Generated watermark sequence for {args.nums}: {wm_seq}")

	# Reshape to (4, 4)
	message_info = np.array(wm_seq).reshape(4, 4)

	# 2. Initialize Template Generator
	generator = WM_Template_Generator(
		ratio_u= args.ratio_u,  # 红绿通道
		ratio_v= args.ratio_v,  # 蓝黄通道
		type = args.type_val,
		v_tp_fn=chosen_pattern_fn,
	)
	###### Get watermark template images positve/ inverse template ######
	wm_images = generator.wm_template_imgs
	wm_images_inverse = generator.wm_template_imgs_inverse

	# 3. Calculate dimensions
	block_width = args.screen_width // args.block_cols
	block_height = args.screen_height // args.block_rows
	message_width = block_width // MESSAGE_COL
	message_height = block_height // MESSAGE_ROW

	# 4. Generate watermark images
	wm_blank, wm_blank_inverse = gen_watermark_imgs(
		message_info,
		args.block_rows,
		args.block_cols,
		block_height,
		block_width,
		message_height,
		message_width,
		wm_images,
		wm_images_inverse,
	)

	# 5. Save images
	os.makedirs(args.save_dir, exist_ok=True)

	wm_resized = resize_template_image(wm_blank, args.screen_width, args.screen_height)
	base_name = f"wm_template_{args.nums}"
	save_path = os.path.join(args.save_dir, f"{base_name}.png")
	cv2.imwrite(save_path, wm_resized)

	wm_resized_inverse = resize_template_image(wm_blank_inverse, args.screen_width, args.screen_height)
	save_path_inverse = os.path.join(args.save_dir, f"{base_name}_inverse.png")
	cv2.imwrite(save_path_inverse, wm_resized_inverse)


	if args.train is not None:
		print(f"Train mode enabled with codeword {args.train}")
	print(f"Saved channel-data templates (R=Y, G=Cr dynamic, B=Cb static) to {args.save_dir}: {base_name}.png / {base_name}_inverse.png")


