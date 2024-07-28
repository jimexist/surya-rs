FROM rust:1.80-slim as builder

ENV OPEN_CV_VERSION="4.10.0"

RUN apt-get update && apt-get install -y \
  build-essential \
  clang \
  libclang-dev \
  libssl-dev \
  wget \
  zip \
  cmake

WORKDIR /usr/src/opencv

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/${OPEN_CV_VERSION}.zip && \
  unzip opencv.zip && \
  rm opencv.zip

RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPEN_CV_VERSION}.zip && \
  unzip opencv_contrib.zip && \
  rm opencv_contrib.zip

WORKDIR /usr/src/opencv/build

RUN cmake -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=NO \
  -DCMAKE_INSTALL_PREFIX=/opt/opencv \
  -DBUILD_DOCS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_ITT=OFF \
  -DBUILD_IPP_IW=OFF \
  -DWITH_PNG=OFF \
  -DWITH_JPEG=OFF \
  -DWITH_TIFF=OFF \
  -DWITH_WEBP=OFF \
  -DWITH_OPENJPEG=OFF \
  -DWITH_JASPER=OFF \
  -DWITH_OPENEXR=OFF \
  -DWITH_V4L=OFF \
  -DWITH_CAROTENE=OFF \
  -DBUILD_opencv_java=OFF \
  -DBUILD_opencv_python=OFF \
  -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPEN_CV_VERSION}/modules \
  ../opencv-${OPEN_CV_VERSION}

RUN cmake --build . --target install --config Release --parallel 8

RUN cmake --install . --prefix /opt/opencv

WORKDIR /usr/src/surya

COPY . .

RUN OPENCV_LINK_LIBS="opencv_imgcodecs,opencv_imgproc,opencv_core" \
  OPENCV_LINK_PATHS="/opt/opencv/lib,/opt/opencv/lib/opencv4/3rdparty,/usr/lib/$(uname -m)-linux-gnu" \
  OPENCV_INCLUDE_PATHS="/opt/opencv/include,/opt/opencv/include/opencv4" \
  OPENSSL_LIB_DIR="/usr/lib/$(uname -m)-linux-gnu" \
  OPENSSL_INCLUDE_DIR="/usr/include/openssl" \
  cargo install --path . --features "cli"

FROM debian:bookworm-slim

RUN apt-get update && \
  apt-get install -y libssl-dev && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/bin

COPY --from=builder /usr/local/cargo/bin/surya /usr/local/bin/surya

ENTRYPOINT ["surya"]
