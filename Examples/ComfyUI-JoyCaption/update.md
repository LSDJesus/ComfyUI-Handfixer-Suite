# ComfyUI-JoyCaption Update Log


## V1.1.1 (2025-06-07)
### Bug Fixes
- Fixed CaptionTool nodes not registering in ComfyUI interface

### Internationalization (i18n)
![v1 1 1](https://github.com/user-attachments/assets/bcb6cadd-1294-4fd0-a5b4-fe8cd3784801)

- Added multi-language support for node interfaces
  - English (en) - Default language
  - French (fr) - Support for French interface
  - Japanese (ja) - 日本語インターフェース対応
  - Korean (ko) - 한국어 인터페이스 지원
  - Russian (ru) - Поддержка русского интерфейса
  - Chinese (zh) - 中文界面支持
 
## V1.1.0 (2025-06-05)
### Features
- Initial release of ComfyUI-JoyCaption
- Added JoyCaption node for image captioning
- Integrated memory management system
- Added caption tools for text processing

[![Joycaption_node](example_workflows/batch_image_text_output.jpg)](https://github.com/1038lab/ComfyUI-JoyCaption/blob/main/example_workflows/batch_image_text_output.json)

![Batch Caption](https://github.com/user-attachments/assets/2e03348d-213e-4a49-b303-375ff129f66d)

### Memory Management
- Implemented efficient memory handling for large image processing
- Added automatic memory cleanup after processing
- Optimized memory usage during batch operations
- Added memory usage monitoring

### Caption Tools
- Added Image Batch Path node (🖼️) for batch image loading
  - Support for sequential, reverse, and random image loading
  - Configurable batch size and start position
  - Automatic EXIF orientation correction
  - Support for jpg, jpeg, png, and webp formats
- Added Caption Saver node (📝) for caption management
  - Flexible output path configuration
  - Custom filename support
  - Optional image copying with captions
  - Automatic file overwrite protection
  - UTF-8 encoding support
  - Batch processing capability
