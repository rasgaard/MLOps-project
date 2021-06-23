torch-model-archiver -f --model-name "FakeNews" \
                        --version 1.0 \
                        --serialized-file ./models/roberta_fakenews-final/pytorch_model.bin \
                        --extra-files "./models/roberta_fakenews-final/config.json,./src/models/deployment/index_to_name.json" \
                        --handler "./src/models/deployment/handler.py" \
                        --export-path "./models/" && \
torchserve --start --model-store models --models FakeNews=FakeNews.mar --ncs