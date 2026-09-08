FROM rust:1-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
RUN cargo build --release --locked && mkdir -p /data

FROM gcr.io/distroless/cc-debian13
COPY --from=build --chown=65534:65534 /src/target/release/clipd /clipd
COPY --chown=65534:65534 models/vision.onnx models/text.onnx models/tokenizer.json /models/
COPY --from=build --chown=65534:65534 /data /data
USER 65534:65534
ENV DATA_DIR=/data MODEL_DIR=/models PORT=8080
EXPOSE 8080
ENTRYPOINT ["/clipd"]
