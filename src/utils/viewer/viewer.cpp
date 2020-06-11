// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <stdlib.h>

#include <Magnum/configure.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#ifdef MAGNUM_TARGET_WEBGL
#include <Magnum/Platform/EmscriptenApplication.h>
#else
#include <Magnum/Platform/GlfwApplication.h>
#endif
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/Timeline.h>

#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Shaders/Shaders.h>

#include "esp/assets/ResourceManager.h"
#include "esp/gfx/RenderCamera.h"
#include "esp/nav/PathFinder.h"
#include "esp/physics/PhysicsManager.h"
#include "esp/physics/RigidObject.h"
#include "esp/scene/ObjectControls.h"
#include "esp/scene/SceneManager.h"
#include "esp/scene/SceneNode.h"

#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/Directory.h>
#include <Corrade/Utility/String.h>
#include <Magnum/DebugTools/Screenshot.h>
#include <Magnum/EigenIntegration/GeometryIntegration.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <sophus/so3.hpp>
#include "esp/core/Utility.h"
#include "esp/core/esp.h"
#include "esp/gfx/Drawable.h"
#include "esp/io/io.h"

#include "esp/scene/SceneConfiguration.h"
#include "esp/sim/Simulator.h"

#include "esp/gfx/configure.h"

constexpr float moveSensitivity = 0.1f;
constexpr float lookSensitivity = 11.25f;
constexpr float rgbSensorHeight = 1.5f;
// for ease of access
namespace Cr = Corrade;
namespace Mn = Magnum;

namespace {

using namespace Mn::Math::Literals;

class Viewer : public Mn::Platform::Application {
 public:
  explicit Viewer(const Arguments& arguments);

 private:
  void drawEvent() override;
  void viewportEvent(ViewportEvent& event) override;
  void mousePressEvent(MouseEvent& event) override;
  void mouseReleaseEvent(MouseEvent& event) override;
  void mouseMoveEvent(MouseMoveEvent& event) override;
  void mouseScrollEvent(MouseScrollEvent& event) override;
  void keyPressEvent(KeyEvent& event) override;
  void keyReleaseEvent(KeyEvent& event) override;
  void updateRenderCamera();

  // Interactive functions
  void addObject(const std::string& configHandle);
  void addObject(int objID);

  // add template-derived object
  void addTemplateObject();

  // add primiitive object
  void addPrimitiveObject();

  void pokeLastObject();
  void pushLastObject();
  void torqueLastObject();
  void removeLastObject();
  void wiggleLastObject();

  void recomputeNavMesh(const std::string& sceneFilename,
                        esp::nav::NavMeshSettings& navMeshSettings);

  void invertGravity();
  Mn::Vector3 randomDirection();

  void toggleNavMeshVisualization();

  Mn::Vector3 positionOnSphere(Mn::SceneGraph::Camera3D& camera,
                               const Mn::Vector2i& position);

  // single inline for logging agent state msgs, so can be easily modified
  inline void logAgentStateMsg(bool showPos, bool showOrient) {
    std::stringstream strDat("");
    if (showPos) {
      strDat << "Agent position "
             << Eigen::Map<esp::vec3f>(agentBodyNode_->translation().data())
             << " ";
    }
    if (showOrient) {
      strDat << "Agent orientation "
             << esp::quatf(agentBodyNode_->rotation()).coeffs().transpose();
    }

    auto str = strDat.str();
    if (str.size() > 0) {
      LOG(INFO) << str;
    }
  }

  esp::assets::ResourceManager resourceManager_;
  // SceneManager must be before physicsManager_ as the physicsManager_
  // assumes that it "owns" things owned by the scene manager
  esp::scene::SceneManager sceneManager_;

  std::shared_ptr<esp::physics::PhysicsManager> physicsManager_;

  bool debugBullet_ = false;

  std::vector<int> sceneID_;
  esp::scene::SceneNode* agentBodyNode_ = nullptr;
  esp::scene::SceneNode* rgbSensorNode_ = nullptr;

  esp::scene::SceneNode* navSceneNode_ = nullptr;

  esp::scene::SceneGraph* sceneGraph_;
  esp::scene::SceneNode* rootNode_;

  esp::scene::SceneNode* navmeshVisNode_ = nullptr;

  esp::gfx::RenderCamera* renderCamera_ = nullptr;
  esp::nav::PathFinder::ptr pathfinder_;
  esp::scene::ObjectControls controls_;
  std::vector<int> objectIDs_;

  bool drawObjectBBs = false;

  Mn::Timeline timeline_;

  Mn::ImGuiIntegration::Context imgui_{Mn::NoCreate};
  bool showFPS_ = true;
  bool frustumCullingEnabled_ = true;

  // framebuffer for drawable selection
  Mn::GL::Framebuffer selectionFramebuffer_{Mn::NoCreate};
  Mn::GL::Renderbuffer selectionDepth_;
  Mn::GL::Renderbuffer selectionDrawableID_;
  bool objectSelectionOn_ = false;
};

Viewer::Viewer(const Arguments& arguments)
    : Mn::Platform::Application{arguments,
                                Configuration{}
                                    .setTitle("Viewer")
                                    .setWindowFlags(
                                        Configuration::WindowFlag::Resizable),
                                GLConfiguration{}
                                    .setColorBufferSize(
                                        Mn::Vector4i(8, 8, 8, 8))
                                    .setSampleCount(4)},
      pathfinder_(esp::nav::PathFinder::create()),
      controls_() {
  Cr::Utility::Arguments args;
#ifdef CORRADE_TARGET_EMSCRIPTEN
  args.addNamedArgument("scene")
#else
  args.addArgument("scene")
#endif
      .setHelp("scene", "scene file to load")
      .addSkippedPrefix("magnum", "engine-specific options")
      .setGlobalHelp("Displays a 3D scene file provided on command line")
      .addBooleanOption("enable-physics")
      .addBooleanOption("scene-requires-lighting")
      .setHelp("scene-requires-lighting", "scene requires lighting")
      .addBooleanOption("debug-bullet")
      .setHelp("debug-bullet", "render Bullet physics debug wireframes")
      .addOption("physics-config", ESP_DEFAULT_PHYS_SCENE_CONFIG)
      .setHelp("physics-config", "physics scene config file")
      .addOption("navmesh-file")
      .setHelp("navmesh-file", "manual override path to scene navmesh file")
      .addBooleanOption("recompute-navmesh")
      .setHelp("recompute-navmesh", "programmatically generate scene navmesh")
      .parse(arguments.argc, arguments.argv);

  const auto viewportSize = Mn::GL::defaultFramebuffer.viewport().size();

  imgui_ =
      Mn::ImGuiIntegration::Context(Mn::Vector2{windowSize()} / dpiScaling(),
                                    windowSize(), framebufferSize());

  /* Set up proper blending to be used by ImGui. There's a great chance
     you'll need this exact behavior for the rest of your scene. If not, set
     this only for the drawFrame() call. */
  Mn::GL::Renderer::setBlendEquation(Mn::GL::Renderer::BlendEquation::Add,
                                     Mn::GL::Renderer::BlendEquation::Add);
  Mn::GL::Renderer::setBlendFunction(
      Mn::GL::Renderer::BlendFunction::SourceAlpha,
      Mn::GL::Renderer::BlendFunction::OneMinusSourceAlpha);

  // Setup renderer and shader defaults
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::DepthTest);
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::FaceCulling);

  int sceneID = sceneManager_.initSceneGraph();
  sceneID_.push_back(sceneID);
  sceneGraph_ = &sceneManager_.getSceneGraph(sceneID);
  rootNode_ = &sceneGraph_->getRootNode();
  navSceneNode_ = &rootNode_->createChild();

  auto& drawables = sceneGraph_->getDrawables();
  const std::string& file = args.value("scene");
  esp::assets::AssetInfo info = esp::assets::AssetInfo::fromPath(file);
  std::string sceneLightSetup = esp::assets::ResourceManager::NO_LIGHT_KEY;
  if (args.isSet("scene-requires-lighting")) {
    info.requiresLighting = true;
    sceneLightSetup = esp::assets::ResourceManager::DEFAULT_LIGHTING_KEY;
  }

  if (args.isSet("enable-physics")) {
    std::string physicsConfigFilename = args.value("physics-config");
    if (!Cr::Utility::Directory::exists(physicsConfigFilename)) {
      LOG(FATAL)
          << physicsConfigFilename
          << " was not found, specify an existing file in --physics-config";
    }
    if (!resourceManager_.loadScene(info, physicsManager_, navSceneNode_,
                                    &drawables, sceneLightSetup,
                                    physicsConfigFilename)) {
      LOG(FATAL) << "cannot load " << file;
    }
    if (args.isSet("debug-bullet")) {
      debugBullet_ = true;
    }
  } else {
    if (!resourceManager_.loadScene(info, navSceneNode_, &drawables,
                                    sceneLightSetup)) {
      LOG(FATAL) << "cannot load " << file;
    }
  }

  const Mn::Range3D& sceneBB = rootNode_->computeCumulativeBB();
  resourceManager_.setLightSetup(esp::gfx::getLightsAtBoxCorners(sceneBB));

  // Set up camera
  renderCamera_ = &sceneGraph_->getDefaultRenderCamera();
  agentBodyNode_ = &rootNode_->createChild();
  rgbSensorNode_ = &agentBodyNode_->createChild();

  rgbSensorNode_->translate({0.0f, rgbSensorHeight, 0.0f});
  agentBodyNode_->translate({0.0f, 0.0f, 5.0f});

  renderCamera_->setProjectionMatrix(viewportSize.x(),  // width
                                     viewportSize.y(),  // height
                                     0.01f,             // znear
                                     1000.0f,           // zfar
                                     90.0f);            // hfov
  renderCamera_->setAspectRatioPolicy(
      Mn::SceneGraph::AspectRatioPolicy::Extend);

  // Load navmesh if available
  std::string navmeshFilename;
  if (!args.value("navmesh-file").empty()) {
    navmeshFilename = Corrade::Utility::Directory::join(
        Corrade::Utility::Directory::current(), args.value("navmesh-file"));
  } else if (file.compare(esp::assets::EMPTY_SCENE)) {
    navmeshFilename = esp::io::changeExtension(file, ".navmesh");

    // TODO: short term solution to mitigate issue #430
    // we load the pre-computed navmesh for the ptex mesh to avoid
    // online computation.
    // for long term solution, see issue #430
    if (Cr::Utility::String::endsWith(file, "mesh.ply")) {
      navmeshFilename = Corrade::Utility::Directory::join(
          Corrade::Utility::Directory::path(file) + "/habitat",
          "mesh_semantic.navmesh");
    }
  }

  if (esp::io::exists(navmeshFilename) && !args.isSet("recompute-navmesh")) {
    LOG(INFO) << "Loading navmesh from " << navmeshFilename;
    pathfinder_->loadNavMesh(navmeshFilename);
  } else if (file.compare(esp::assets::EMPTY_SCENE)) {
    esp::nav::NavMeshSettings navMeshSettings;
    navMeshSettings.setDefaults();
    recomputeNavMesh(file, navMeshSettings);
  }

  // connect controls to navmesh if loaded
  if (pathfinder_->isLoaded()) {
    // some scenes could have pathable roof polygons. We are not filtering
    // those starting points here.
    esp::vec3f position = pathfinder_->getRandomNavigablePoint();
    agentBodyNode_->setTranslation(Mn::Vector3(position));

    controls_.setMoveFilterFunction([&](const esp::vec3f& start,
                                        const esp::vec3f& end) {
      esp::vec3f currentPosition = pathfinder_->tryStep(start, end);
      LOG(INFO) << "position=" << currentPosition.transpose() << " rotation="
                << esp::quatf(agentBodyNode_->rotation()).coeffs().transpose();
      LOG(INFO) << "Distance to closest obstacle: "
                << pathfinder_->distanceToClosestObstacle(currentPosition);

      return currentPosition;
    });
  }

  renderCamera_->node().setTransformation(
      rgbSensorNode_->absoluteTransformation());

  // setup an offscreen frame buffer for object selection
  selectionDepth_.setStorage(Mn::GL::RenderbufferFormat::DepthComponent24,
                             framebufferSize());
  selectionDrawableID_.setStorage(Mn::GL::RenderbufferFormat::R16UI,
                                  framebufferSize());
  selectionFramebuffer_ = Mn::GL::Framebuffer{{{}, framebufferSize()}};
  selectionFramebuffer_
      .attachRenderbuffer(Mn::GL::Framebuffer::BufferAttachment::Depth,
                          selectionDepth_)
      .attachRenderbuffer(Mn::GL::Framebuffer::ColorAttachment{1},
                          selectionDrawableID_);
  selectionFramebuffer_.mapForDraw({{Mn::Shaders::Generic3D::ColorOutput,
                                     Mn::GL::Framebuffer::DrawAttachment::None},
                                    {Mn::Shaders::Generic3D::ObjectIdOutput,
                                     Mn::GL::Framebuffer::ColorAttachment{1}}});
  CORRADE_INTERNAL_ASSERT(
      selectionFramebuffer_.checkStatus(Mn::GL::FramebufferTarget::Draw) ==
      Mn::GL::Framebuffer::Status::Complete);

  timeline_.start();

}  // end Viewer::Viewer

void Viewer::addObject(int ID) {
  if (physicsManager_ == nullptr)
    return;
  addObject(resourceManager_.getPhysicsObjectTemplateHandle(ID));
}  // addObject

void Viewer::addObject(const std::string& configFile) {
  if (physicsManager_ == nullptr)
    return;

  Mn::Matrix4 T =
      agentBodyNode_
          ->MagnumObject::transformationMatrix();  // Relative to agent bodynode
  Mn::Vector3 new_pos = T.transformPoint({0.1f, 1.5f, -2.0f});

  auto& drawables = sceneGraph_->getDrawables();

  int physObjectID = physicsManager_->addObject(configFile, &drawables);
  physicsManager_->setTranslation(physObjectID, new_pos);

  physicsManager_->setRotation(physObjectID, esp::core::randomRotation());

  objectIDs_.push_back(physObjectID);

}  // addObject

// add file-based template derived object from keypress
void Viewer::addTemplateObject() {
  if (physicsManager_ != nullptr) {
    int numObjTemplates = resourceManager_.getNumFileTemplateObjects();
    if (numObjTemplates > 0) {
      addObject(resourceManager_.getRandomFileTemplateHandle());
    } else
      LOG(WARNING) << "No objects loaded, can't add any";
  } else
    LOG(WARNING) << "Run the app with --enable-physics in order to add "
                    "templated-based physically modeled objects";
}  // addTemplateObject

// add synthesized primiitive object from keypress
void Viewer::addPrimitiveObject() {
  // TODO : use this to implement synthesizing rendered physical objects
  if (physicsManager_ != nullptr) {
    int numObjPrims = resourceManager_.getNumSynthTemplateObjects();
    if (numObjPrims > 0) {
      addObject(resourceManager_.getRandomSynthTemplateHandle());
    } else
      LOG(WARNING) << "No primitive templates available, can't add any objects";
  } else
    LOG(WARNING) << "Run the app with --enable-physics in order to add "
                    "physically modelled primitives";
}  // addPrimitiveObject

void Viewer::removeLastObject() {
  if (physicsManager_ == nullptr || objectIDs_.size() == 0)
    return;
  physicsManager_->removeObject(objectIDs_.back());
  objectIDs_.pop_back();
}

void Viewer::invertGravity() {
  if (physicsManager_ == nullptr)
    return;
  const Mn::Vector3& gravity = physicsManager_->getGravity();
  const Mn::Vector3 invGravity = -1 * gravity;
  physicsManager_->setGravity(invGravity);
}

void Viewer::pokeLastObject() {
  if (physicsManager_ == nullptr || objectIDs_.size() == 0)
    return;
  Mn::Matrix4 T =
      agentBodyNode_
          ->MagnumObject::transformationMatrix();  // Relative to agent bodynode
  Mn::Vector3 impulse = T.transformVector({0.0f, 0.0f, -3.0f});
  Mn::Vector3 rel_pos = Mn::Vector3(0.0f, 0.0f, 0.0f);
  physicsManager_->applyImpulse(objectIDs_.back(), impulse, rel_pos);
}

void Viewer::pushLastObject() {
  if (physicsManager_ == nullptr || objectIDs_.size() == 0)
    return;
  Mn::Matrix4 T =
      agentBodyNode_
          ->MagnumObject::transformationMatrix();  // Relative to agent bodynode
  Mn::Vector3 force = T.transformVector({0.0f, 0.0f, -40.0f});
  Mn::Vector3 rel_pos = Mn::Vector3(0.0f, 0.0f, 0.0f);
  physicsManager_->applyForce(objectIDs_.back(), force, rel_pos);
}

void Viewer::recomputeNavMesh(const std::string& sceneFilename,
                              esp::nav::NavMeshSettings& navMeshSettings) {
  esp::nav::PathFinder::ptr pf = esp::nav::PathFinder::create();

  esp::assets::MeshData::uptr joinedMesh =
      resourceManager_.createJoinedCollisionMesh(sceneFilename);

  if (!pf->build(navMeshSettings, *joinedMesh)) {
    LOG(ERROR) << "Failed to build navmesh";
    return;
  }

  LOG(INFO) << "reconstruct navmesh successful";
  pathfinder_ = pf;
}

void Viewer::torqueLastObject() {
  if (physicsManager_ == nullptr || objectIDs_.size() == 0)
    return;
  Mn::Vector3 torque = randomDirection() * 30;
  physicsManager_->applyTorque(objectIDs_.back(), torque);
}

// generate random direction vectors:
Mn::Vector3 Viewer::randomDirection() {
  Mn::Vector3 dir(1.0f, 1.0f, 1.0f);
  while (sqrt(dir.dot()) > 1.0) {
    dir = Mn::Vector3((float)((rand() % 2000 - 1000) / 1000.0),
                      (float)((rand() % 2000 - 1000) / 1000.0),
                      (float)((rand() % 2000 - 1000) / 1000.0));
  }
  dir = dir / sqrt(dir.dot());
  return dir;
}

void Viewer::wiggleLastObject() {
  // demo of kinematic motion capability
  // randomly translate last added object
  if (physicsManager_ == nullptr || objectIDs_.size() == 0)
    return;

  Mn::Vector3 randDir = randomDirection();
  // Only allow +Y so dynamic objects don't push through the floor.
  randDir[1] = abs(randDir[1]);

  physicsManager_->translate(objectIDs_.back(), randDir * 0.1);
}

void Viewer::toggleNavMeshVisualization() {
  if (navmeshVisNode_ == nullptr && pathfinder_->isLoaded()) {
    // test navmesh visualization
    navmeshVisNode_ = &rootNode_->createChild();
    int nevMeshVisPrimID = resourceManager_.loadNavMeshVisualization(
        *pathfinder_, navmeshVisNode_, &sceneGraph_->getDrawables());
    navmeshVisNode_->translate({0, 0.1, 0});
  } else if (navmeshVisNode_ != nullptr) {
    delete navmeshVisNode_;
    navmeshVisNode_ = nullptr;
  }
}

Mn::Vector3 Viewer::positionOnSphere(Mn::SceneGraph::Camera3D& camera,
                                     const Mn::Vector2i& position) {
  // Convert from window to frame coordinates.
  Mn::Vector2 framePosition =
      (Mn::Vector2{position} * Mn::Vector2{framebufferSize()}) /
      Mn::Vector2{windowSize()};
  const Mn::Vector2 positionNormalized =
      framePosition / Mn::Vector2{camera.viewport()} - Mn::Vector2{0.5f};
  const Mn::Float length = positionNormalized.length();
  const Mn::Vector3 result(
      length > 1.0f ? Mn::Vector3(positionNormalized, 0.0f)
                    : Mn::Vector3(positionNormalized, 1.0f - length));
  return (result * Mn::Vector3::yScale(-1.0f)).normalized();
}

float timeSinceLastSimulation = 0.0;
void Viewer::drawEvent() {
  Mn::GL::defaultFramebuffer.clear(Mn::GL::FramebufferClear::Color |
                                   Mn::GL::FramebufferClear::Depth);
  if (sceneID_.size() <= 0)
    return;

  if (physicsManager_ != nullptr)
    // step physics at a fixed rate
    timeSinceLastSimulation += timeline_.previousFrameDuration();
  if (timeSinceLastSimulation >= 1.0 / 60.0) {
    physicsManager_->stepPhysics(1.0 / 60.0);
    timeSinceLastSimulation = 0.0;
  }

  int DEFAULT_SCENE = 0;
  int sceneID = sceneID_[DEFAULT_SCENE];
  auto& sceneGraph = sceneManager_.getSceneGraph(sceneID);
  uint32_t visibles = 0;

  for (auto& it : sceneGraph.getDrawableGroups()) {
    // TODO: remove || true
    if (it.second.prepareForDraw(*renderCamera_) || true) {
      visibles += renderCamera_->draw(it.second, frustumCullingEnabled_);
    }
  }

  if (debugBullet_) {
    Mn::Matrix4 camM(renderCamera_->cameraMatrix());
    Mn::Matrix4 projM(renderCamera_->projectionMatrix());

    physicsManager_->debugDraw(projM * camM);
  }

  imgui_.newFrame();

  if (showFPS_) {
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::Begin("main", NULL,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
                     ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::SetWindowFontScale(2.0);
    ImGui::Text("%.1f FPS", Mn::Double(ImGui::GetIO().Framerate));
    uint32_t total = sceneGraph.getDrawables().size();
    ImGui::Text("%u drawables", total);
    ImGui::Text("%u culled", total - visibles);
    ImGui::End();
  }

  /* Set appropriate states. If you only draw ImGui, it is sufficient to
     just enable blending and scissor test in the constructor. */
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::Blending);
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::ScissorTest);
  Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::FaceCulling);
  Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::DepthTest);

  imgui_.drawFrame();

  /* Reset state. Only needed if you want to draw something else with
     different state after. */
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::DepthTest);
  Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::FaceCulling);
  Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::ScissorTest);
  Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::Blending);

  swapBuffers();
  timeline_.nextFrame();
  redraw();
}

void Viewer::viewportEvent(ViewportEvent& event) {
  Mn::GL::defaultFramebuffer.setViewport({{}, framebufferSize()});
  renderCamera_->setViewport(event.windowSize());
  imgui_.relayout(Mn::Vector2{event.windowSize()} / event.dpiScaling(),
                  event.windowSize(), event.framebufferSize());
}

void Viewer::mousePressEvent(MouseEvent& event) {
  if (event.button() == MouseEvent::Button::Right && objectSelectionOn_) {
    selectionFramebuffer_
        .bind(); /** @todo mapForDraw() should bind implicitly */
    selectionFramebuffer_
        .mapForDraw({{Mn::Shaders::Generic3D::ColorOutput,
                      Mn::GL::Framebuffer::DrawAttachment::None},
                     {Mn::Shaders::Generic3D::ObjectIdOutput,
                      Mn::GL::Framebuffer::ColorAttachment{1}}})
        .clearDepth(1.0f)
        .clearColor(1, Mn::Vector4ui{0xffff});
    CORRADE_INTERNAL_ASSERT(
        selectionFramebuffer_.checkStatus(Mn::GL::FramebufferTarget::Draw) ==
        Mn::GL::Framebuffer::Status::Complete);
  }
  event.setAccepted();
}

void Viewer::mouseReleaseEvent(MouseEvent& event) {
  event.setAccepted();
}

void Viewer::mouseScrollEvent(MouseScrollEvent& event) {
  if (!event.offset().y()) {
    return;
  }

  /* Distance to origin */
  const float distance =
      renderCamera_->node().transformation().translation().z();

  /* Move 15% of the distance back or forward */
  controls_(*agentBodyNode_, "moveForward",
            distance * (1.0f - (event.offset().y() > 0 ? 1 / 0.85f : 0.85f)));

  logAgentStateMsg(true, true);
  updateRenderCamera();
  redraw();

  event.setAccepted();
}

void Viewer::mouseMoveEvent(MouseMoveEvent& event) {
  if (!(event.buttons() & MouseMoveEvent::Button::Left)) {
    return;
  }
  const Mn::Vector2i delta = event.relativePosition();
  controls_(*agentBodyNode_, "turnRight", delta.x());
  controls_(*rgbSensorNode_, "lookDown", delta.y(), false);

  logAgentStateMsg(true, true);
  updateRenderCamera();
  redraw();

  event.setAccepted();
}

void Viewer::keyPressEvent(KeyEvent& event) {
  const auto key = event.key();
  bool agentMoved = false;
  switch (key) {
    case KeyEvent::Key::Esc:
      std::exit(0);
      break;
    case KeyEvent::Key::LeftShift:
      objectSelectionOn_ = true;
      break;
    case KeyEvent::Key::Left:
      controls_(*agentBodyNode_, "turnLeft", lookSensitivity);
      agentMoved = true;
      break;
    case KeyEvent::Key::Right:
      controls_(*agentBodyNode_, "turnRight", lookSensitivity);
      agentMoved = true;
      break;
    case KeyEvent::Key::Up:
      controls_(*rgbSensorNode_, "lookUp", lookSensitivity, false);
      agentMoved = true;
      break;
    case KeyEvent::Key::Down:
      controls_(*rgbSensorNode_, "lookDown", lookSensitivity, false);
      agentMoved = true;
      break;
    case KeyEvent::Key::Eight:
      addPrimitiveObject();
      break;
    case KeyEvent::Key::Nine:
      if (pathfinder_->isLoaded()) {
        const esp::vec3f position = pathfinder_->getRandomNavigablePoint();
        agentBodyNode_->setTranslation(Mn::Vector3(position));
      }
      break;
    case KeyEvent::Key::A:
      controls_(*agentBodyNode_, "moveLeft", moveSensitivity);
      agentMoved = true;
      break;
    case KeyEvent::Key::D:
      controls_(*agentBodyNode_, "moveRight", moveSensitivity);
      agentMoved = true;
      break;
    case KeyEvent::Key::S:
      controls_(*agentBodyNode_, "moveBackward", moveSensitivity);
      agentMoved = true;
      break;
    case KeyEvent::Key::W:
      controls_(*agentBodyNode_, "moveForward", moveSensitivity);
      agentMoved = true;
      break;
    case KeyEvent::Key::X:
      controls_(*agentBodyNode_, "moveDown", moveSensitivity, false);
      agentMoved = true;
      break;
    case KeyEvent::Key::Z:
      controls_(*agentBodyNode_, "moveUp", moveSensitivity, false);
      agentMoved = true;
      break;
    case KeyEvent::Key::E:
      frustumCullingEnabled_ ^= true;
      break;
    case KeyEvent::Key::C:
      showFPS_ = !showFPS_;
      break;
    case KeyEvent::Key::O:
      addTemplateObject();
      break;
    case KeyEvent::Key::P:
      pokeLastObject();
      break;
    case KeyEvent::Key::F:
      pushLastObject();
      break;
    case KeyEvent::Key::K:
      wiggleLastObject();
      break;
    case KeyEvent::Key::U:
      removeLastObject();
      break;
    case KeyEvent::Key::V:
      invertGravity();
      break;
    case KeyEvent::Key::T:
      // Test key. Put what you want here...
      torqueLastObject();
      break;
    case KeyEvent::Key::N:
      toggleNavMeshVisualization();
      break;
    case KeyEvent::Key::I:
      Mn::DebugTools::screenshot(Mn::GL::defaultFramebuffer,
                                 "test_image_save.png");
      break;
    case KeyEvent::Key::B: {
      // toggle bounding box on objects
      drawObjectBBs = !drawObjectBBs;
      for (auto id : physicsManager_->getExistingObjectIDs()) {
        physicsManager_->setObjectBBDraw(id, &sceneGraph_->getDrawables(),
                                         drawObjectBBs);
      }
    } break;
    default:
      break;
  }
  if (agentMoved) {
    logAgentStateMsg(true, true);
  }
  updateRenderCamera();
  redraw();
}

void Viewer::updateRenderCamera() {
  renderCamera_->node().setTransformation(
      rgbSensorNode_->absoluteTransformation());
}

void Viewer::keyReleaseEvent(KeyEvent& event) {
  const auto key = event.key();
  switch (key) {
    case KeyEvent::Key::LeftShift:
      objectSelectionOn_ = false;
      break;
  }
}

}  // namespace

MAGNUM_APPLICATION_MAIN(Viewer)
