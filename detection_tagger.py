import cv2
import os
import sys
import json
import numpy as np
import argparse
import distinctipy


top_left = ()
bottom_right = ()
mouse_down = False
window_name = '| Detection Tagger |'


IMAGES_EXTENSIONS = ('png', 'jpeg', 'jpg', 'bmp')


class Data:
	def __init__(self, name, images_to_boxes, images_to_scales):
		self.name = name
		self.__images_to_boxes = images_to_boxes
		self.__images_to_scales = images_to_scales
		self.__actions = []
		self.__actions_pointer = -1

	def add(self, image_path, bounding_box):
		true_bounding_box = self.rescale_bounding_box(image_path, bounding_box)
		self.__images_to_boxes[image_path].append(true_bounding_box)
		self.__actions = self.__actions[:self.__actions_pointer + 1] + [('add', [true_bounding_box])]
		self.__actions_pointer += 1

	def remove(self, image_path, x, y):
		x, y = self.__rescale_point(image_path, x, y)
		keep_boxes, del_boxes = [], []
		for ibb, bb in enumerate(self.__images_to_boxes[image_path]):
			if not (bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3]):
				keep_boxes.append(bb)
			else:
				del_boxes.append(bb)
		self.__images_to_boxes[image_path] = keep_boxes
		self.__actions = self.__actions[: self.__actions_pointer + 1] + [('remove', del_boxes)]
		self.__actions_pointer += 1

	def clear(self, image_path):
		self.__actions = self.__actions[:self.__actions_pointer + 1] + [('clear', self.__images_to_boxes[image_path])]

	def __do(self, image_path, undo=True, redo=False):
		assert (undo and not redo) or (not undo and redo)
		if redo:
			self.__actions_pointer += 1
		if len(self.__actions) > 0 and ((redo and (self.__actions_pointer < len(self.__actions))) or (undo and self.__actions_pointer >= 0)):
			action, items = self.__actions[self.__actions_pointer]
			if (action == 'add' and undo) or (action in ['remove', 'clear'] and redo):
				for item in items:
					self.__images_to_boxes[image_path].remove(item)
			elif (action == 'add' and redo) or (action in ['remove', 'clear'] and undo):
				self.__images_to_boxes[image_path] += items
			if undo:
				self.__actions_pointer -= 1
		self.__actions_pointer = min(max(-1, self.__actions_pointer), len(self.__actions) - 1)

	def undo(self, image_path):
		self.__do(image_path, undo=True, redo=False)

	def redo(self, image_path):
		self.__do(image_path, undo=False, redo=True)

	def get_boxes(self, image_path=None):
		if image_path is not None:
			return self.__images_to_boxes[image_path]
		return self.__images_to_boxes

	def get_resize(self, image_path):
		if image_path is not None:
			return self.__images_to_scales[image_path]['dst']
		return self.__images_to_scales

	def rescale_bounding_box(self, image_path, bounding_box):
		top_left_bb = self.__rescale_point(image_path, bounding_box[0], bounding_box[1])
		bottom_right_bb = self.__rescale_point(image_path, bounding_box[2], bounding_box[3])
		bounding_box = top_left_bb + bottom_right_bb
		return bounding_box

	def scale_bouding_box(self, image_path, true_bounding_box):
		top_left_bb = self.__scale_point(image_path, true_bounding_box[0], true_bounding_box[1])
		bottom_right_bb = self.__scale_point(image_path, true_bounding_box[2], true_bounding_box[3])
		bounding_box = top_left_bb + bottom_right_bb
		return bounding_box

	def __rescale_point(self, image_path, x, y):
		(h1, w1), (h2, w2) = self.__images_to_scales[image_path]['src'], self.__images_to_scales[image_path]['dst']
		true_x = int(np.round((float(w1) / float(w2)) * x))
		true_y = int(np.round((float(h1) / float(h2)) * y))
		return true_x, true_y

	def __scale_point(self, image_path, true_x, true_y):
		(h1, w1), (h2, w2) = self.__images_to_scales[image_path]['dst'], self.__images_to_scales[image_path]['src']
		x = int(np.round((float(w1) / float(w2)) * true_x))
		y = int(np.round((float(h1) / float(h2)) * true_y))
		return x, y


def load_tags(folder_path, classes, max_height, max_width):
	print('Loading Project...')
	labels = {}
	image_paths = [os.path.join(folder_path, fn) for fn in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, fn)) and fn.endswith(IMAGES_EXTENSIONS)]
	metadata_file = os.path.join(folder_path, 'metadata.json')
	if os.path.isfile(metadata_file):
		with open(metadata_file, mode='r', encoding='utf-8') as mf:
			done_images = json.load(mf)
	else:
		done_images = {}
	for image_path in image_paths:
		if image_path not in done_images.keys():
			done_images[image_path] = 0
	job_is_done = sum(done_images.values()) == len(image_paths)
	if not job_is_done:
		for c in classes:
			images_to_boxes = {}
			images_to_scales = {}
			for image_path in image_paths:
				images_to_scales[image_path] = {}
				if os.path.isfile(image_path + '.json'):
					with open(image_path + '.json', mode='r', encoding='utf-8') as imf:
						labeled_data = json.load(imf)
						if c in labeled_data.keys():
							images_to_boxes[image_path] = labeled_data[c]
						else:
							images_to_boxes[image_path] = []
				else:
					images_to_boxes[image_path] = []
				image = cv2.imread(image_path)
				height, width = (image.shape[0], image.shape[1])
				images_to_scales[image_path]['src'] = (height, width)
				if height > max_height:
					width = int((float(max_height) / float(height)) * width)
					height = max_height
				if width > max_width:
					height = int((float(max_width) / float(width)) * height)
					width = max_width
				images_to_scales[image_path]['dst'] = (height, width)
			labels[c] = Data(c, images_to_boxes, images_to_scales)
	else:
		labels = {}
	return labels, image_paths, done_images, job_is_done


def save_tags(image_path, labels, done_images=None):
	folder_path = '{}'.format(os.sep).join(image_path.split(os.sep)[:-1])
	metadata_file = os.path.join(folder_path, 'metadata.json')
	if done_images is not None:
		with open(metadata_file, mode='w') as mf:
			json.dump(done_images, mf)
	with open(image_path + '.json', mode='w') as imf:
		json.dump({c: labels[c].get_boxes(image_path) for c in labels.keys()}, imf)


def open_image(image_path, labels, current_class, class_color):
	image = cv2.imread(image_path)
	cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)
	cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  
	(dst_height, dst_width) = labels[current_class].get_resize(image_path)
	image = cv2.resize(image, (dst_width, dst_height), interpolation=cv2.INTER_AREA)
	for bounding_box in labels[current_class].get_boxes(image_path):
		scaled_bouding_box = labels[current_class].scale_bouding_box(image_path, bounding_box)
		cv2.rectangle(image, (scaled_bouding_box[0], scaled_bouding_box[1]), (scaled_bouding_box[2], scaled_bouding_box[3]), class_color, 1, 8)
	cv2.imshow(window_name, image)
	cv2.setWindowTitle(window_name, window_name + ' Current Class : ' + current_class)
	return image


def drawBoundingBox(action, x, y, flags, *userdata):
	labels = userdata[0][0]
	image_path = userdata[0][1]
	current_class = userdata[0][2]
	class_color = userdata[0][3]
	global top_left, bottom_right, mouse_down
	if action == cv2.EVENT_LBUTTONDOWN:
		top_left = (x, y)
		mouse_down = True
	elif action == cv2.EVENT_LBUTTONUP:
		bottom_right = (x, y)
		mouse_down = False
		top_left_x, top_left_y = min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])
		bottom_right_x, bottom_right_y = max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])
		top_left = (top_left_x, top_left_y)
		bottom_right = (bottom_right_x, bottom_right_y)
		if top_left[0] != bottom_right[0] or top_left[1] != bottom_right[1]:
			labels[current_class].add(image_path, [top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
			open_image(image_path, labels, current_class, class_color)
			save_tags(image_path, labels)
	elif action == cv2.EVENT_LBUTTONDBLCLK:
		labels[current_class].remove(image_path, x, y)
		open_image(image_path, labels, current_class, class_color)
		save_tags(image_path, labels)
	elif action == cv2.EVENT_MOUSEMOVE and mouse_down:
		bottom_right = (x, y)
		image = open_image(image_path, labels, current_class, class_color)
		cv2.rectangle(image, top_left, bottom_right, class_color, 1, 8)
		cv2.imshow(window_name, image)


def tag_image(image_path, labels, current_class, class_color):
	k = 0
	while k not in [113, 97, 100, 32, 13]:
		open_image(image_path, labels, current_class, class_color)
		cv2.setMouseCallback(window_name, drawBoundingBox, (labels, image_path, current_class, class_color))
		k = cv2.waitKey(0)
		if k == 99:
			labels[current_class].clear(image_path)
		elif k == 121:
			labels[current_class].redo(image_path)
		elif k == 122:
			labels[current_class].undo(image_path)
		save_tags(image_path, labels)
	return k


def tag_folder(folder_path, classes, max_height=950, max_width=1750):
	labels, image_paths, done_images, job_is_done = load_tags(folder_path, classes, max_height, max_width)
	if job_is_done:
		return
	colors = distinctipy.get_colors(len(classes))
	classes_colors = [(int(i[0] * 255), int(i[1] * 255), int(i[2] * 255)) for i in colors]
	current_class_idx = 0
	current_class = classes[0]
	done = [done_images[p] for p in image_paths]
	min_idx_to_show = done.index(0)
	done.reverse()
	max_idx_to_show = (len(done) - 1) - done.index(0)
	image_idx = min_idx_to_show
	print('Current Class : {}'.format(current_class))
	while True:
		q = tag_image(image_paths[image_idx], labels, current_class, classes_colors[current_class_idx])
		if q == 100:
			image_idx += 1
		elif q == 97:
			image_idx -= 1
		elif q == 13:
			done_images[image_paths[image_idx]] = 1
			save_tags(image_paths[image_idx], labels, done_images)
			if sum(done.values()) == len(image_paths):
				break
			done = [done_images[p] for p in image_paths]
			min_idx_to_show = done.index(0)
			done.reverse()
			max_idx_to_show = (len(done) - 1) - done.index(0)
			image_idx += 1
			print('Progress : {}/{}'.format(sum(done_images.values()), len(image_paths)))
		elif q == 32:
			current_class_idx = (current_class_idx + 1) % len(classes)
			current_class = classes[current_class_idx]
			print('Current Class : {}'.format(current_class))
		elif q == 113:
			break
		image_idx = max(image_idx, min_idx_to_show)
		image_idx = min(image_idx, max_idx_to_show)
		while done_images[image_paths[image_idx]] == 1:
			if q == 97:
				image_idx = (image_idx - 1) % len(image_paths)
			else:
				image_idx = (image_idx + 1) % len(image_paths)
	cv2.destroyAllWindows()


def get_parser():
	parser = argparse.ArgumentParser(description='Detection Tagger')
	parser.add_argument('folder_path', type=str, help='Folder containing images to label')
	parser.add_argument('classes', type=str, help='String seperated by \',\' for each class')
	parser.add_argument('--max_height', type=int, default=950, help='Max height')
	parser.add_argument('--max_width', type=int, default=1750, help='Max width')
	return parser


if __name__ == '__main__':
	parser = get_parser()
	args = parser.parse_args()
	tag_folder(args.folder_path, args.classes.split(','), args.max_height, args.max_width)

