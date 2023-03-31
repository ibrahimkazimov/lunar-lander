import { VELOCITY_MULTIPLIER, GRAVITY } from "./constants.js";

export const generateCanvas = ({ width, height, attachNode }) => {
  const element = document.createElement("canvas");
  const context = element.getContext("2d");

  element.style.width = width + "px";
  element.style.height = height + "px";

  const scale = window.devicePixelRatio;
  element.width = Math.floor(width * scale);
  element.height = Math.floor(height * scale);
  context.scale(scale, scale);

  document.querySelector(attachNode).appendChild(element);

  return [context, width, height, element, scale];
};

export const animate = (drawFunc) => {
  let startTime = Date.now();
  let currentFrameTime = Date.now();

  const resetStartTime = () => (startTime = Date.now());

  const drawFuncContainer = () => {
    currentFrameTime = Date.now();
    drawFunc(currentFrameTime - startTime);
    window.requestAnimationFrame(drawFuncContainer);
  };

  window.requestAnimationFrame(drawFuncContainer);

  return { resetStartTime };
};

export const randomBool = (probability = 0.5) => Math.random() >= probability;

export const randomBetween = (min, max) => Math.random() * (max - min) + min;

export const seededRandomBetween = (min, max, seededRandom) =>
  seededRandom.getSeededRandom() * (max - min) + min;

export const seededRandomBool = (seededRandom, probability = 0.5) =>
  seededRandom.getSeededRandom() >= probability;

export const getVectorVelocity = (velocity) =>
  Math.sqrt(Math.pow(velocity.x, 2) + Math.pow(velocity.y, 2));

export const getAngleDeltaUpright = (angle) => {
  const angleInDeg = (angle * 180) / Math.PI;
  const repeatingAngle = Math.abs(angleInDeg) % 360;
  return repeatingAngle > 180 ? Math.abs(repeatingAngle - 360) : repeatingAngle;
};

export const getAngleDeltaUprightWithSign = (angle) => {
  const angleInDeg = (angle * 180) / Math.PI;
  const repeatingAngle = Math.abs(angleInDeg) % 360;
  return repeatingAngle > 180 ? repeatingAngle - 360 : repeatingAngle;
};

export const velocityInMPH = (velocity) =>
  Intl.NumberFormat().format(
    (getVectorVelocity(velocity) * VELOCITY_MULTIPLIER).toFixed(1)
  );

export const heightInFeet = (yPos, groundedHeight) =>
  Intl.NumberFormat().format(
    Math.abs(Math.round((yPos - groundedHeight) / 3.5))
  );

// Progress
// Transforms any number range into a range of 0–1
//
// Expected behavior:
// progress(5, 30, 17.5) -> .5
// progress(30, 5, 17.5) -> .5
// progress(5, 30, 30)   -> 1
export const progress = (start, end, current) =>
  (current - start) / (end - start);

export const percentProgress = (start, end, current) =>
  Math.max(0, Math.min(((current - start) / (end - start)) * 100, 100));

export const isAboveTerrain = (CTX, position, terrain, scaleFactor) => {
  const terrainLandingData = terrain.getLandingData();

  return (
    position.y < terrainLandingData.terrainHeight ||
    (position.y >= terrainLandingData.terrainHeight &&
      !CTX.isPointInPath(
        terrainLandingData.terrainPath2D,
        position.x * scaleFactor,
        position.y * scaleFactor
      ))
  );
};

export const getLineAngle = (startCoordinate, endCoordinate) => {
  const dy = endCoordinate.y - startCoordinate.y;
  const dx = endCoordinate.x - startCoordinate.x;
  let theta = Math.atan2(dy, dx);
  theta *= 180 / Math.PI;
  return theta;
};

export const angleReflect = (incidenceAngle, surfaceAngle) => {
  const a = surfaceAngle * 2 - incidenceAngle;
  return a >= 360 ? a - 360 : a < 0 ? a + 360 : a;
};

export const simpleBallisticUpdate = (
  state,
  currentPosition,
  currentVelocity,
  currentAngle,
  rotationDirection,
  currentRotationVelocity
) => {
  const CTX = state.get("CTX");
  const canvasWidth = state.get("canvasWidth");
  const newPosition = { ...currentPosition };
  const newVelocity = { ...currentVelocity };
  let newRotationVelocity;
  let newAngle = currentAngle;

  newPosition.y = currentPosition.y + currentVelocity.y;
  newPosition.x = currentPosition.x + currentVelocity.x;

  if (
    isAboveTerrain(
      CTX,
      newPosition,
      state.get("terrain"),
      state.get("scaleFactor")
    )
  ) {
    newRotationVelocity = rotationDirection
      ? currentRotationVelocity + randomBetween(0, 0.01)
      : currentRotationVelocity - randomBetween(0, 0.01);
    newAngle = currentAngle + (Math.PI / 180) * newRotationVelocity;
    newVelocity.y = currentVelocity.y + GRAVITY;
  } else {
    // Generate the angle reflection of the current vector
    const terrainAngle = state.get("terrain").getSegmentAngleAtX(newPosition.x);
    const currentAngle = getLineAngle(
      { x: 0, y: 0 },
      { x: currentVelocity.x, y: currentVelocity.y }
    );
    const newAngle = angleReflect(currentAngle, terrainAngle);

    // Generate a new velocity moving in the direction of newAngle
    // Slow the velocity down as well
    newVelocity.x =
      Math.cos((newAngle * Math.PI) / 180) / randomBetween(1.5, 3);
    newVelocity.y =
      Math.sin((newAngle * Math.PI) / 180) / randomBetween(1.5, 3);

    // Recalculate position so we don't get stuck in the terrain
    newPosition.y = currentPosition.y + currentVelocity.y;
    newPosition.x = currentPosition.x + currentVelocity.x;

    // Slow down rotation on impact
    newRotationVelocity = currentRotationVelocity / 2;
  }

  if (newPosition.x < 0) newPosition.x = canvasWidth;
  if (newPosition.x > canvasWidth) newPosition.x = 0;

  return [newPosition, newVelocity, newRotationVelocity, newAngle];
};

export const seededShuffleArray = (array, seededRandom) => {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(seededRandom.getSeededRandom() * (i + 1));
    const temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
};
