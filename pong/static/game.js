const isTraining = window.location.pathname.indexOf('train') !== -1

const socket = io.connect('http://' + document.domain + ':' + location.port);

function getInitialVelocity() {
  const xVelocity = -1 * (Math.random() * (5 - 3) + 3);
  const yVelocity = Math.sqrt(50 - (xVelocity * xVelocity));
  return [xVelocity, Math.random() > 0.5 ? yVelocity : yVelocity * -1]
}

class Ball {
  constructor(position = [Game.DIM_X / 2, (Game.DIM_Y + Paddle.WIDTH + Paddle.SPACE_FROM_WALL) / 2], velocity = getInitialVelocity(), radius = 5) {
    this.position = position;
    this.velocity = velocity;
    this.radius = radius;
  }

  checkCollisions(maxX, maxY, paddle1, paddle2) {
    if (!isTraining) {
      if (this.position[1] >= maxY - (this.radius/ 2) || this.position[1] <= 0 + (this.radius / 2)) {
        this.velocity[1] = -1 * this.velocity[1];
      }
      if (this.position[0] <= 0 + (this.radius / 2)) {
        return 'PASSED_PADDLE';
      }
      if (this.position[0] > Paddle.SPACE_FROM_WALL && this.position[0] < Paddle.SPACE_FROM_WALL + Paddle.WIDTH + (this.radius / 2) && this.position[1] >= paddle1.position && this.position[1] <= paddle1.position + Paddle.HEIGHT) {
        this.velocity[0] = -1 * this.velocity[0];
        return 'GOOD_RETURN';
      }
      if (this.position[0] >= maxX - (this.radius / 2)) {
        return 'PASSED_PADDLE';
      }
      if (this.position[0] < maxX - Paddle.SPACE_FROM_WALL && this.position[0] > maxX - (Paddle.SPACE_FROM_WALL + Paddle.WIDTH + (this.radius / 2)) && this.position[1] >= paddle2.position && this.position[1] <= paddle2.position + Paddle.HEIGHT) {
        this.velocity[0] = -1 * this.velocity[0];
        return 'GOOD_RETURN';
      }
    } else {
      if (paddle1.side === 'left') {
        if (this.position[0] <= 0 + (this.radius / 2)) {
          return 'PASSED_PADDLE';
        }
        if (this.position[0] >= maxX - (this.radius / 2)) {
          this.velocity[0] = -1 * this.velocity[0];
        }
        if (this.position[1] >= maxY - (this.radius/ 2) || this.position[1] <= 0 + (this.radius / 2)) {
          this.velocity[1] = -1 * this.velocity[1];
        }
        if (this.position[0] > Paddle.SPACE_FROM_WALL && this.position[0] < Paddle.SPACE_FROM_WALL + Paddle.WIDTH + (this.radius / 2) && this.position[1] >= paddle1.position && this.position[1] <= paddle1.position + Paddle.HEIGHT) {
          this.velocity[0] = -1 * this.velocity[0];
          return 'GOOD_RETURN';
        }
      } else {
        if (this.position[0] <= 0 + (this.radius / 2) + Paddle.WIDTH + Paddle.SPACE_FROM_WALL) {
          this.velocity[0] = -1 * this.velocity[0];
        }
        if (this.position[0] >= maxX - (this.radius / 2)) {
          return 'PASSED_PADDLE';
        }
        if (this.position[1] >= maxY - (this.radius/ 2) || this.position[1] <= 0 + (this.radius / 2)) {
          this.velocity[1] = -1 * this.velocity[1];
        }
        if (this.position[0] < maxX - Paddle.SPACE_FROM_WALL && this.position[0] > maxX - (Paddle.SPACE_FROM_WALL + Paddle.WIDTH + (this.radius / 2)) && this.position[1] >= paddle1.position && this.position[1] <= paddle1.position + Paddle.HEIGHT) {
          this.velocity[0] = -1 * this.velocity[0];
          return 'GOOD_RETURN';
        }
      }
    }
  }

  render(ctx) {
    ctx.fillStyle = 'white';
    ctx.beginPath();

    ctx.arc(
      this.position[0],
      this.position[1],
      this.radius,
      0,
      2 * Math.PI,
      false
    );

    ctx.fill();
  }

  move() {
    this.position = [this.position[0] + this.velocity[0], this.position[1] + this.velocity[1]];
  }
}

class Paddle {
  constructor(position = (Game.DIM_Y - Paddle.HEIGHT) / 2, velocity = 5, side = 'right') {
    this.position = position;
    this.velocity = velocity;
    this.xPosition = side === 'left' ? Paddle.SPACE_FROM_WALL : Game.DIM_X - Paddle.SPACE_FROM_WALL - Paddle.WIDTH;
    this.side = side;
  }

  render(ctx) {
    ctx.fillStyle = 'white';
    ctx.fillRect(this.xPosition, this.position, Paddle.WIDTH, Paddle.HEIGHT);
  }

  move() {
    if (!(this.position <= 0 && this.velocity < 0) && !(this.position >= Game.DIM_Y - Paddle.HEIGHT && this.velocity > 0)) {
      this.position = this.position + this.velocity;
    }
  }

  moveUp() {
    this.velocity = -5;
  }

  moveDown() {
    this.velocity = 5;
  }

  stop() {
    this.velocity = 0;
  }
}

Paddle.HEIGHT = 35;
Paddle.WIDTH = 10;
Paddle.SPACE_FROM_WALL = 15;

class Game {
  constructor(ctx) {
    this.ctx = ctx;
    this.ball = new Ball();
    if (!isTraining) {
      this.paddle1 = new Paddle(undefined, 0, 'left');
      this.paddle2 = new Paddle(undefined, 0);
    } else {
      this.paddle1 = new Paddle()
    }
  }

  render() {
    this.ctx.clearRect(0, 0, Game.DIM_X, Game.DIM_Y);
    this.ctx.fillStyle = Game.BG_COLOR;
    this.ctx.fillRect(0, 0, Game.DIM_X, Game.DIM_Y);
    if (isTraining) {
      this.ctx.fillStyle = 'white';
      this.ctx.fillRect(0, 0, Paddle.WIDTH + Paddle.SPACE_FROM_WALL, Game.DIM_Y);
    }
    this.ball.render(this.ctx);
    this.paddle1.render(this.ctx);
    if (this.paddle2) {
      this.paddle2.render(this.ctx);
    }
  }

  checkCollisions() {
    return this.ball.checkCollisions(Game.DIM_X, Game.DIM_Y, this.paddle1, this.paddle2);
  }

  bindKeyHandlers() {
    window.onkeydown = e => {
      if (isTraining) {
        if (e.keyCode === 38) {
          this.paddle1.moveUp();
        }
      } else {
        if (e.keyCode === 40) {
          this.paddle1.moveDown();
        }
        if (e.keyCode === 38) {
          this.paddle1.moveUp();
        }
      }
    }

    window.onkeyup = e => {
      if (isTraining) {
        if (e.keyCode === 38) {
          this.paddle1.moveDown();
        }
      } else {
        if (e.keyCode === 40 || e.keyCode === 38) {
          this.paddle1.stop();
        }
      }
    };
  }

  resetBall() {
    this.ball = new Ball()
  }

  getCurrentData() {
    return {
      paddle_velocity: isTraining ? this.paddle1.velocity : this.paddle2.velocity,
      paddle_position: isTraining ? this.paddle1.velocity : this.paddle2.position,
      ball_x_velocity: this.ball.velocity[0],
      ball_y_velocity: this.ball.velocity[1],
      ball_x_position: this.ball.position[0],
      ball_y_position: this.ball.position[1],
    }
  }

  handleData(data) {
    fetch('/api/game_data_binary', {
      method: 'POST',
      body: JSON.stringify(data),
      headers: { 'Content-Type': 'application/json' },
    })
  }

  start() {
    this.bindKeyHandlers();
    let data = [];

    const animateCallback = () => {
      this.ball.move();
      this.paddle1.move();
      if (this.paddle2) {
        this.paddle2.move();
      }
      const collisionType = this.checkCollisions();
      if (collisionType) {
        if (collisionType === 'PASSED_PADDLE') {
          setTimeout(() => this.resetBall(), 500);
          data = [];
        }
        if (collisionType === 'GOOD_RETURN') {
          if (isTraining) {
            this.handleData(data);
          }
          data = [];
        }
      }
      const currentData = this.getCurrentData()
      if (isTraining) {
        data.push(currentData);
      } else {
        socket.emit('current data', this.getCurrentData())
      }
      this.render();
      requestAnimationFrame(animateCallback);
    };

    animateCallback();
  }
}

Game.DIM_X = 600;
Game.DIM_Y = 400;
Game.BG_COLOR = 'black';

document.addEventListener("DOMContentLoaded", function(){
  const canvasEl = document.getElementsByTagName("canvas")[0];
  canvasEl.width = Game.DIM_X;
  canvasEl.height = Game.DIM_Y;

  const ctx = canvasEl.getContext("2d");
  game = new Game(ctx);
  game.start();
  socket.on('move', data => {
    if (data.move === 0) {
      game.paddle2.moveUp();
    } else if (data.move === 1) {
      game.paddle2.moveDown();
    }
  })
});
