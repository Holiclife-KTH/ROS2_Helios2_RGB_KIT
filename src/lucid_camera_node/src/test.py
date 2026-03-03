import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from custom_msgs.msg import BoundingBoxMultiArray
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import threading


class ObjectPositionEstimator(Node):
    def __init__(self):
        super().__init__("object_position_estimator")

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # calibration 캐시
        self._camera_matrix = None
        self._dist_coeffs = None
        self._rvec = None
        self._tvec = None

        # 최신 PointCloud 캐시 (투영 결과 포함)
        self._cached_points_xyz = None
        self._cached_projected_2d = None
        self._cached_pcd_header = None

        # 구독자
        self.pcd_sub = self.create_subscription(
            PointCloud2, "helios/pointcloud_rgb", self.pcd_callback, 10
        )

        self.image_sub = self.create_subscription(
            Image, "triton/image_raw", self.image_callback, 10
        )

        self.bbox_sub = self.create_subscription(
            BoundingBoxMultiArray,
            "real_time_segmentation_node/segmented_bbox",
            self.bbox_callback,
            10,
        )

        # 퍼블리셔
        self.position_pub = self.create_publisher(PointStamped, "object/position", 10)

        # 디버그용 퍼블리셔
        self.marker_pub = self.create_publisher(MarkerArray, "object/bbox_markers", 10)
        self.roi_pcd_pub = self.create_publisher(
            PointCloud2, "object/roi_pointcloud", 10
        )
        self.debug_pcd_pub = self.create_publisher(
            PointCloud2, "object/debug_projected", 10
        )

        # Triton 이미지 해상도 (bbox 클램핑용)
        self.image_width = 2048
        self.image_height = 1536

        self.get_logger().info("ObjectPositionEstimator started")

    def pcd_callback(self, msg: PointCloud2):
        """PointCloud 수신 시 3D→2D 투영을 미리 계산하여 캐시"""
        points_xyz = self.pointcloud2_to_numpy(msg)
        if points_xyz is None or len(points_xyz) == 0:
            return

        projected_2d = self.project_3d_to_2d(points_xyz)
        if projected_2d is None:
            return

        with self.lock:
            self._cached_points_xyz = points_xyz
            self._cached_projected_2d = projected_2d
            self._cached_pcd_header = msg.header

    def image_callback(self, msg: Image):
        """이미지 해상도 자동 감지"""
        self.image_width = msg.width
        self.image_height = msg.height

    def bbox_callback(self, msg: BoundingBoxMultiArray):
        """bbox 수신 시 캐시된 투영 결과를 사용하여 위치 추정 + 3D bbox 시각화"""
        with self.lock:
            points_xyz = self._cached_points_xyz
            projected_2d = self._cached_projected_2d
            pcd_header = self._cached_pcd_header

        if points_xyz is None or projected_2d is None:
            self.get_logger().warn(
                "No PointCloud data cached yet", throttle_duration_sec=2.0
            )
            return

        marker_array = MarkerArray()
        # 이전 마커 삭제
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for idx, box in enumerate(msg.data):
            # bbox 좌표 추출 및 클램핑
            x1 = max(0, int(box.bbox[0]))
            y1 = max(0, int(box.bbox[1]))
            x2 = min(self.image_width - 1, int(box.bbox[2]))
            y2 = min(self.image_height - 1, int(box.bbox[3]))

            cls_name = box.cls
            conf = box.conf

            if x2 <= x1 or y2 <= y1:
                continue

            # bbox 내부의 3D 포인트 추출
            roi_points, roi_mask = self.get_roi_points(
                points_xyz, projected_2d, [x1, y1, x2, y2]
            )

            if roi_points is None or len(roi_points) < 3:
                self.get_logger().warn(
                    f"[{cls_name}] No valid points in bbox [{x1},{y1},{x2},{y2}]"
                )
                continue

            # ROI 포인트클라우드 publish (디버그)
            self.publish_roi_pointcloud(roi_points, pcd_header)

            # Outlier 제거
            clean_points = self.remove_outliers_iqr(roi_points)
            if len(clean_points) < 3:
                clean_points = roi_points

            # 중심 좌표 계산
            centroid = np.median(clean_points, axis=0)

            # 3D bounding box 마커 생성
            bbox_marker = self.create_3d_bbox_marker(
                clean_points, centroid, pcd_header, idx, cls_name, conf
            )
            marker_array.markers.append(bbox_marker)

            # 중심점 마커
            center_marker = self.create_center_marker(
                centroid, pcd_header, idx + 1000, cls_name
            )
            marker_array.markers.append(center_marker)

            # 텍스트 마커
            text_marker = self.create_text_marker(
                centroid, pcd_header, idx + 2000, cls_name, conf, len(clean_points)
            )
            marker_array.markers.append(text_marker)

            # 위치 publish
            position = (centroid[0], centroid[1], centroid[2], len(clean_points))
            self.publish_position(position, pcd_header, cls_name)

            self.get_logger().info(
                f"[{cls_name}] (conf={conf:.2f}): "
                f"X={centroid[0]:.3f}m, Y={centroid[1]:.3f}m, Z={centroid[2]:.3f}m "
                f"(points: {len(clean_points)}/{len(roi_points)})"
            )

        # 마커 퍼블리시
        self.marker_pub.publish(marker_array)

    def get_roi_points(self, points_xyz, projected_2d, bbox):
        """
        bbox 내부의 3D 포인트와 마스크를 반환

        Returns:
            (valid_points, mask) 또는 (None, None)
        """
        x1, y1, x2, y2 = bbox

        mask_bbox = (
            (projected_2d[:, 0] >= x1)
            & (projected_2d[:, 0] <= x2)
            & (projected_2d[:, 1] >= y1)
            & (projected_2d[:, 1] <= y2)
        )

        roi_points = points_xyz[mask_bbox]

        if len(roi_points) == 0:
            return None, None

        # 유효성 검사
        valid_mask = (
            ~np.isnan(roi_points[:, 0])
            & ~np.isnan(roi_points[:, 1])
            & ~np.isnan(roi_points[:, 2])
            & (roi_points[:, 2] > 0.05)
            & (roi_points[:, 2] < 3.0)
        )
        valid_points = roi_points[valid_mask]

        if len(valid_points) < 3:
            return None, None

        return valid_points, mask_bbox

    def create_3d_bbox_marker(
        self, points, centroid, header, marker_id, cls_name, conf
    ):
        """
        ROI 포인트들의 min/max로 3D bounding box (CUBE) 마커 생성
        """
        marker = Marker()
        marker.header = header
        marker.ns = "bbox_3d"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # 포인트들의 min/max로 bbox 크기 결정
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)

        # 중심 위치
        marker.pose.position.x = float((min_pt[0] + max_pt[0]) / 2.0)
        marker.pose.position.y = float((min_pt[1] + max_pt[1]) / 2.0)
        marker.pose.position.z = float((min_pt[2] + max_pt[2]) / 2.0)
        marker.pose.orientation.w = 1.0

        # 크기 (최소 1cm)
        marker.scale.x = max(float(max_pt[0] - min_pt[0]), 0.01)
        marker.scale.y = max(float(max_pt[1] - min_pt[1]), 0.01)
        marker.scale.z = max(float(max_pt[2] - min_pt[2]), 0.01)

        # 색상 (반투명 녹색)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.3

        marker.lifetime.sec = 1
        marker.lifetime.nanosec = 0

        return marker

    def create_center_marker(self, centroid, header, marker_id, cls_name):
        """중심점 구체 마커"""
        marker = Marker()
        marker.header = header
        marker.ns = "center_point"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(centroid[0])
        marker.pose.position.y = float(centroid[1])
        marker.pose.position.z = float(centroid[2])
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03

        # 빨간색
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime.sec = 1
        marker.lifetime.nanosec = 0

        return marker

    def create_text_marker(self, centroid, header, marker_id, cls_name, conf, n_points):
        """텍스트 라벨 마커"""
        marker = Marker()
        marker.header = header
        marker.ns = "text_label"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(centroid[0])
        marker.pose.position.y = float(centroid[1])
        marker.pose.position.z = float(centroid[2]) + 0.05  # 약간 위에 표시
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.03  # 텍스트 크기

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = (
            f"{cls_name} ({conf:.2f})\n"
            f"X:{centroid[0]:.3f} Y:{centroid[1]:.3f} Z:{centroid[2]:.3f}\n"
            f"pts: {n_points}"
        )

        marker.lifetime.sec = 1
        marker.lifetime.nanosec = 0

        return marker

    def publish_roi_pointcloud(self, roi_points, header):
        """ROI 내부 포인트만 별도 토픽으로 publish (디버그용)"""

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(roi_points)
        msg.is_dense = False
        msg.is_bigendian = False

        # 필드 정의 (x, y, z)
        from sensor_msgs.msg import PointField

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = 12 * len(roi_points)
        msg.data = roi_points.astype(np.float32).tobytes()

        self.roi_pcd_pub.publish(msg)

    def pointcloud2_to_numpy(self, pcd_msg: PointCloud2) -> np.ndarray:
        """PointCloud2 메시지를 (N, 3) numpy 배열로 고속 변환"""

        field_offsets = {}
        for field in pcd_msg.fields:
            field_offsets[field.name] = field.offset

        if not all(k in field_offsets for k in ("x", "y", "z")):
            self.get_logger().error("PointCloud2 missing x/y/z fields")
            return None

        point_step = pcd_msg.point_step
        n_points = pcd_msg.width * pcd_msg.height

        data = np.frombuffer(pcd_msg.data, dtype=np.uint8)

        if len(data) < n_points * point_step:
            return None

        data = data[: n_points * point_step].reshape(n_points, point_step)

        x_offset = field_offsets["x"]
        y_offset = field_offsets["y"]
        z_offset = field_offsets["z"]

        x = data[:, x_offset : x_offset + 4].copy().view(np.float32).flatten()
        y = data[:, y_offset : y_offset + 4].copy().view(np.float32).flatten()
        z = data[:, z_offset : z_offset + 4].copy().view(np.float32).flatten()

        return np.column_stack((x, y, z))

    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """3D 포인트(미터)를 Triton 카메라 이미지 좌표로 재투영"""
        if self._camera_matrix is None:
            self._load_calibration()

        # mm 단위로 변환 (lucid_node에서 /1000 했으므로)
        points_mm = points_3d * 1000.0

        projected, _ = cv2.projectPoints(
            points_mm.astype(np.float64),
            self._rvec,
            self._tvec,
            self._camera_matrix,
            self._dist_coeffs,
        )

        return projected.reshape(-1, 2)

    def _load_calibration(self):
        """orientation.yml에서 calibration 파라미터 로드"""

        cal_file = "/home/irol/workspace/ros2_helios2_rgb_kit/src/lucid_camera_node/resource/orientation.yml"

        fs = cv2.FileStorage(cal_file, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            self.get_logger().error(f"Failed to open calibration file: {cal_file}")
            raise RuntimeError("Calibration file not found")

        self._camera_matrix = fs.getNode("cameraMatrix").mat()
        self._dist_coeffs = fs.getNode("distCoeffs").mat()
        self._rvec = fs.getNode("rotationVector").mat()
        self._tvec = fs.getNode("translationVector").mat()

        fs.release()
        self.get_logger().info(f"Loaded calibration from {cal_file}")

    def remove_outliers_iqr(self, points: np.ndarray) -> np.ndarray:
        """IQR 방법으로 Z축 기준 outlier 제거"""

        z_values = points[:, 2]
        q1 = np.percentile(z_values, 25)
        q3 = np.percentile(z_values, 75)
        iqr = q3 - q1

        if iqr < 0.001:
            return points

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (z_values >= lower) & (z_values <= upper)
        return points[mask]

    def publish_position(self, position, header: Header, cls_name: str = ""):
        """추정된 3D 위치를 PointStamped로 퍼블리시"""

        msg = PointStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = header.frame_id
        msg.point.x = float(position[0])
        msg.point.y = float(position[1])
        msg.point.z = float(position[2])

        self.position_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectPositionEstimator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
