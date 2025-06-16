const { createCanvas, loadImage } = require('canvas');

class SpriteRenderer {
  constructor(options = {}) {
    this.frameSize = options.frameSize || 64;
    this.sheetWidth = options.sheetWidth || 832;
    this.sheetHeight = options.sheetHeight || 3456;
    
    // Create main canvas
    this.canvas = createCanvas(this.sheetWidth, this.sheetHeight);
    this.ctx = this.canvas.getContext('2d');
    
    // Animation definitions
    this.baseAnimations = {
      spellcast: 0,
      thrust: 4 * this.frameSize,
      walk: 8 * this.frameSize,
      slash: 12 * this.frameSize,
      shoot: 16 * this.frameSize,
      hurt: 20 * this.frameSize,
      climb: 21 * this.frameSize,
      idle: 22 * this.frameSize,
      jump: 26 * this.frameSize,
      sit: 30 * this.frameSize,
      emote: 34 * this.frameSize,
      run: 38 * this.frameSize,
      combat_idle: 42 * this.frameSize,
      backslash: 46 * this.frameSize,
      halfslash: 50 * this.frameSize
    };

    this.animationFrameCounts = {
      spellcast: 7,
      thrust: 8,
      walk: 9,
      slash: 6,
      shoot: 13,
      hurt: 6,
      climb: 6,
      idle: 2,
      jump: 5,
      sit: 3,
      emote: 3,
      run: 8,
      combat_idle: 2,
      backslash: 13,
      halfslash: 7
    };

    // Store loaded images
    this.images = new Map();
  }

  async loadImage(path) {
    if (this.images.has(path)) {
      return this.images.get(path);
    }
    
    try {
      const image = await loadImage(path);
      this.images.set(path, image);
      return image;
    } catch (error) {
      console.error(`Failed to load image: ${path}`, error);
      return null;
    }
  }

  async drawItems(items) {
    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Sort items by z-index
    items.sort((a, b) => parseInt(a.zPos) - parseInt(b.zPos));

    // Draw each item
    for (const item of items) {
      const { fileName, supportedAnimations, customAnimation } = item;

      if (customAnimation) {
        await this.drawCustomAnimation(item);
      } else {
        await this.drawStandardAnimations(item);
      }
    }

    return this.canvas;
  }

  async drawStandardAnimations(item) {
    const { fileName, supportedAnimations } = item;
    const [directory, file] = this.splitFilePath(fileName);

    for (const [animName, yOffset] of Object.entries(this.baseAnimations)) {
      let animationToCheck = animName;
      if (animName === 'combat_idle') {
        animationToCheck = 'combat';
      } else if (animName === 'backslash') {
        animationToCheck = '1h_slash';
      } else if (animName === 'halfslash') {
        animationToCheck = '1h_halfslash';
      }

      if (supportedAnimations.includes(animationToCheck)) {
        const imagePath = `${directory}/${animName}/${file}`;
        const image = await this.loadImage(imagePath);
        if (image) {
          this.ctx.drawImage(image, 0, yOffset);
        }
      }
    }
  }

  async drawCustomAnimation(item) {
    const { fileName, customAnimation } = item;
    const image = await this.loadImage(fileName);
    if (!image) return;

    const customAnimDef = this.customAnimations[customAnimation];
    if (!customAnimDef) return;

    const frameSize = customAnimDef.frameSize;
    const width = frameSize * customAnimDef.frames[0].length;
    const height = frameSize * customAnimDef.frames.length;

    // Create temporary canvas for custom animation
    const tempCanvas = createCanvas(width, height);
    const tempCtx = tempCanvas.getContext('2d');

    // Draw frames
    for (let i = 0; i < customAnimDef.frames.length; i++) {
      const frames = customAnimDef.frames[i];
      for (let j = 0; j < frames.length; j++) {
        const [rowName, x] = frames[j].split(',');
        const y = this.animationRowsLayout[rowName] + 1;
        const offset = (frameSize - this.frameSize) / 2;

        // Get frame from main canvas
        const frameData = this.ctx.getImageData(
          this.frameSize * x,
          this.frameSize * y,
          this.frameSize,
          this.frameSize
        );

        // Draw to temp canvas
        tempCtx.putImageData(
          frameData,
          frameSize * j + offset,
          frameSize * i + offset
        );
      }
    }

    // Draw custom animation to main canvas
    this.ctx.drawImage(tempCanvas, 0, this.sheetHeight);
  }

  splitFilePath(filePath) {
    const index = filePath.lastIndexOf('/');
    if (index > -1) {
      return [
        filePath.substring(0, index),
        filePath.substring(index + 1)
      ];
    }
    throw new Error(`Could not split path: ${filePath}`);
  }

  // Helper method to save the canvas to a file
  async saveToFile(path) {
    const fs = require('fs');
    const out = fs.createWriteStream(path);
    const stream = this.canvas.createPNGStream();
    stream.pipe(out);
    return new Promise((resolve, reject) => {
      out.on('finish', resolve);
      out.on('error', reject);
    });
  }
}

module.exports = SpriteRenderer; 
