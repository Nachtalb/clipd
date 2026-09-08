FROM rust:1-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
RUN cargo build --release --locked

FROM gcr.io/distroless/cc-debian13
COPY --from=build /src/target/release/clipd /clipd
COPY models/vision.onnx models/text.onnx models/tokenizer.json /models/
USER 65534:65534
ENV DATA_DIR=/data MODEL_DIR=/models PORT=8080
EXPOSE 8080
ENTRYPOINT ["/clipd"]
