import pygame as pg
import numpy as np
import moderngl as mgl
import os

from jsbsim_gym.visualization.quaternion import Quaternion # Assuming this path is correct

dir_name = os.path.abspath(os.path.dirname(__file__))

def load_shader(ctx : mgl.Context, vertex_filename, frag_filename):
    with open(os.path.join(dir_name, vertex_filename)) as f:
        vertex_src = f.read()
    with open(os.path.join(dir_name, frag_filename)) as f:
        frag_src = f.read()
    
    return ctx.program(vertex_shader=vertex_src, fragment_shader=frag_src)

def load_mesh(ctx : mgl.Context, program, filename):
    # Check if the filename is absolute or needs to be joined with dir_name
    if not os.path.isabs(filename) and not os.path.exists(filename):
        mesh_file_path = os.path.join(dir_name, filename)
        if not os.path.exists(mesh_file_path):
            # Fallback: try to find it in parent of dir_name if "visualization" is a subdir
            # This is a common pattern if assets are alongside the env files
            parent_dir = os.path.dirname(dir_name)
            fallback_path = os.path.join(parent_dir, filename)
            if os.path.exists(fallback_path):
                mesh_file_path = fallback_path
            else:
                # As a last resort, try relative to CWD (though less reliable)
                if os.path.exists(filename):
                     mesh_file_path = filename
                else:
                    raise FileNotFoundError(f"Mesh file '{filename}' not found in {dir_name}, {parent_dir}, or CWD.")
    elif os.path.exists(filename):
        mesh_file_path = filename
    else:
        raise FileNotFoundError(f"Mesh file '{filename}' (absolute or pre-checked) not found.")

    v = []
    vn = []
    vertices = []
    indices = []

    with open(mesh_file_path, 'r') as file: # Use determined mesh_file_path
        for line in file:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                vertex = [float(val) for val in values[1:4]]
                v.append(vertex)
            elif values[0] == 'vn':
                norm = [float(val) for val in values[1:4]]
                vn.append(norm)
            elif values[0] == 'vt':
                continue
            elif values[0] in ('usemtl', 'usemat'):
                continue
            elif values[0] == 'mtllib':
                continue
            elif values[0] == 'f':
                face_vertices = []
                face_indices_local = []
                for idx, val in enumerate(values[1:]):
                    w = val.split('/')
                    vertex_data = np.hstack((v[int(w[0])-1], vn[int(w[2])-1]))
                    face_vertices.append(vertex_data)
                
                # Triangulate polygon faces (common for OBJ)
                # Assumes convex polygons, usually works for quads by fanning from the first vertex
                start_idx_global = len(vertices) # Before adding new vertices for this face
                for fv_data in face_vertices:
                    vertices.append(fv_data)
                
                for i in range(1, len(face_vertices) - 1):
                    indices.append([start_idx_global, start_idx_global + i, start_idx_global + i + 1])


    if not vertices:
        raise ValueError(f"No vertices loaded from mesh file: {mesh_file_path}")
    if not indices:
        raise ValueError(f"No faces (indices) loaded or generated from mesh file: {mesh_file_path}")

    vbo = ctx.buffer(np.hstack(vertices).astype(np.float32).tobytes())
    ebo = ctx.buffer(np.array(indices).flatten().astype(np.uint32).tobytes()) # Use np.array for indices list
    return ctx.simple_vertex_array(program, vbo, 'aPos', 'aNormal', index_buffer=ebo)


def perspective(fov, aspect, near, far):
    fov *= np.pi/180
    # Corrected perspective matrix calculation (standard OpenGL)
    f = 1.0 / np.tan(fov / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

class Transform:
    def __init__(self):
        self._position = np.zeros(3, dtype=np.float32)
        self._rotation = Quaternion()
        self.scale = 1.0

    @property
    def position(self):
        return self._position.copy()

    @position.setter
    def position(self, position: np.ndarray):
        self._position[:] = position.astype(np.float32)

    @property
    def x(self): return self._position[0]
    @x.setter
    def x(self, x_val: float): self._position[0] = x_val
    @property
    def y(self): return self._position[1]
    @y.setter
    def y(self, y_val: float): self._position[1] = y_val
    @property
    def z(self): return self._position[2]
    @z.setter
    def z(self, z_val: float): self._position[2] = z_val

    @property
    def rotation(self) -> Quaternion:
        return self._rotation.copy()

    @rotation.setter
    def rotation(self, rotation: Quaternion):
        self._rotation._arr[:] = rotation._arr

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = self._rotation.mat().dot(np.eye(3, dtype=np.float32) * self.scale)
        matrix[:3, 3] = self._position
        return matrix

    @property
    def inv_matrix(self) -> np.ndarray:
        # More robust inverse calculation
        # R_inv = R.T (for rotation matrix)
        # T_inv = -R_inv * T (for translation part)
        # S_inv = 1/S
        
        # Inverse of rotation
        rot_inv_mat = self._rotation.inv().mat()
        
        # Inverse of scale
        scale_inv = 1.0 / self.scale if self.scale != 0 else 1.0 # Avoid division by zero
        
        # Combined inverse of rotation and scale for the 3x3 part
        m_3x3_inv = rot_inv_mat * scale_inv
        
        # Inverse of translation part
        pos_inv = -m_3x3_inv.dot(self._position)
        
        inv_m = np.eye(4, dtype=np.float32)
        inv_m[:3, :3] = m_3x3_inv
        inv_m[:3, 3] = pos_inv
        return inv_m

class RenderObject:
    def __init__(self, vao):
        self.vao = vao
        self.color = (1.0, 1.0, 1.0)
        self.transform = Transform()
        self.draw_mode = mgl.TRIANGLES

    def render(self):
        if self.vao is None: return # Skip if no VAO (e.g. Grid before full init)
        self.vao.program['model'] = tuple(np.hstack(self.transform.matrix.T))
        self.vao.program['color'] = self.color
        self.vao.render(self.draw_mode)
class LineSegment(RenderObject):
    def __init__(self, ctx: mgl.Context, program, point_a: np.ndarray, point_b: np.ndarray):
        super().__init__(None) # VAO will be created/updated

        self.ctx = ctx
        self.program = program # Use the same program as the grid (e.g., unlit)
        self.vertices_np = np.zeros((2, 3), dtype=np.float32) # 2 points, 3 coords (x,y,z)
        self.vbo = None # Will be created/updated
        self.update_points(point_a, point_b) # Initial points
        self.draw_mode = mgl.LINES
        self.color = (1.0, 1.0, 0.0) # Yellow for glideslope, for example

    def update_points(self, point_a: np.ndarray, point_b: np.ndarray):
        self.vertices_np[0, :] = point_a.astype(np.float32)
        self.vertices_np[1, :] = point_b.astype(np.float32)

        if self.vbo:
            self.vbo.write(self.vertices_np.tobytes())
        else:
            self.vbo = self.ctx.buffer(self.vertices_np.tobytes())
        
        # VAO needs to be recreated if VBO changes structure or if it's the first time
        # For simplicity here, if VBO is just updated, we might not need to recreate VAO
        # if the shader binding ('aPos') remains the same.
        # However, simple_vertex_array is cheap to call.
        if self.vao: self.vao.release() # Release old VAO before creating new one
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'aPos')


    def render(self): # Override to ensure program is set correctly
        if self.vao is None: return
        # The LineSegment itself is in world coordinates, so its model matrix is identity
        self.program['model'].write(np.eye(4, dtype=np.float32).T.tobytes())
        self.program['color'] = self.color
        self.vao.render(self.draw_mode)
class Grid(RenderObject):
    def __init__(self, ctx : mgl.Context, program, n, spacing):
        super().__init__(None) # VAO set later
        low = -(n-1)*spacing/2
        high = -low
        vertices = []
        indices = []
        for i in range(n):
            vertices.extend([low + spacing*i, 0, low])
            vertices.extend([low + spacing*i, 0,  high])
            indices.extend([i*2, i*2+1])
        for i in range(n):
            vertices.extend([low, 0, low + spacing*i])
            vertices.extend([high, 0, low + spacing*i])
            indices.extend([n*2+i*2, n*2+i*2+1])
        
        if not vertices: # Should not happen with n > 0
            self.vao = None
            return

        vbo = ctx.buffer(np.array(vertices, dtype=np.float32).tobytes())
        ebo = ctx.buffer(np.array(indices, dtype=np.uint32).tobytes())
        self.vao = ctx.simple_vertex_array(program, vbo, 'aPos', index_buffer=ebo)
        self.draw_mode = mgl.LINES

class Viewer:
    def __init__(self, width, height, fps=30, headless=False):
        self.transform = Transform() # Camera transform
        self.width = width
        self.height = height
        self.fps = fps
        self.headless = headless
        self.ctx = None # Initialize to None
        self.offscreen_fbo = None
        self.offscreen_color_att = None
        self.offscreen_depth_att = None
        self.prog = None
        self.unlit = None
        self.objects = []
        self.display = None
        self.clock = None

        try:
            if self.headless:
                self.ctx = mgl.create_standalone_context(require=330)
                # Create an offscreen FBO for headless rendering
                self.offscreen_color_att = self.ctx.texture((self.width, self.height), 4, dtype='f1') # RGBA, f1 for uint8 later
                self.offscreen_depth_att = self.ctx.depth_texture((self.width, self.height))
                self.offscreen_fbo = self.ctx.framebuffer(
                    color_attachments=[self.offscreen_color_att],
                    depth_attachment=self.offscreen_depth_att
                )
            else:
                pg.init()
                pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
                pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
                pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
                pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 1) # For MSAA
                pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, 4) # MSAA samples
                self.display = pg.display.set_mode((width, height), pg.DOUBLEBUF | pg.OPENGL)
                self.ctx = mgl.create_context()
                self.clock = pg.time.Clock()
                self.offscreen_fbo = self.ctx.screen # In windowed mode, render to the screen
            
            self.ctx.enable(mgl.DEPTH_TEST)
            self.ctx.enable(mgl.BLEND) # Optional: for transparency if needed
            self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA # Common blend func
            if not self.headless: self.ctx.enable(mgl.MULTISAMPLE) # Enable MSAA if windowed

            self.projection = perspective(70.0, float(width)/height, 0.1, 1000.0) #fov 70
            
            self.prog = load_shader(self.ctx, "simple.vert", "simple.frag")
            self.prog['projection'].write(self.projection.T.tobytes())
            self.prog['lightDir'].value = (0.8, -0.6, 1.0) # Example light direction

            self.unlit = load_shader(self.ctx, "simple.vert", "unlit.frag") # Assuming simple.vert can be used
            self.unlit['projection'].write(self.projection.T.tobytes())
            
            self.set_view() # Initialize view matrix in shaders

        except Exception as e:
            print(f"Error during Viewer initialization: {e}")
            self.close() # Attempt to clean up if init fails
            raise # Re-raise the exception

    def set_view(self, x=None, y=None, z=None, rotation_quaternion=None):
        if x is not None: self.transform.x = x
        if y is not None: self.transform.y = y
        if z is not None: self.transform.z = z
        if rotation_quaternion is not None: self.transform.rotation = rotation_quaternion
        
        view_matrix_bytes = self.transform.inv_matrix.T.tobytes()
        if self.prog: self.prog['view'].write(view_matrix_bytes)
        if self.unlit: self.unlit['view'].write(view_matrix_bytes)

    def get_frame(self):
        # Read from the FBO (either offscreen_fbo in headless or screen in windowed)
        # Ensure components=3 for RGB if texture is RGBA, or 4 for RGBA
        # Reading as 'f1' (unsigned byte)
        data_bytes = self.offscreen_fbo.read(components=3, alignment=1, dtype='f1')
        img = np.frombuffer(data_bytes, dtype=np.uint8).reshape(self.height, self.width, 3)
        return img[-1::-1,:,:] # Flip vertically (OpenGL origin is bottom-left)

    def render(self):
        if self.ctx is None: return # Not initialized
        
        if not self.headless and self.display:
            pg.event.pump() # Process Pygame events if windowed
        
        self.offscreen_fbo.use() # Activate the target FBO
        self.ctx.clear(0.5, 0.5, 0.5, 1.0) # Clear with alpha=1.0 for the FBO

        for obj in self.objects:
            obj.render() # Renders to the currently active FBO

        if not self.headless and self.display:
            pg.display.flip() # Update the window if not headless
            if self.clock: self.clock.tick(self.fps)

    def close(self):
        # Release ModernGL resources
        for obj in self.objects:
            if obj.vao:
                obj.vao.release()
        if self.prog: self.prog.release()
        if self.unlit: self.unlit.release()
        if self.offscreen_color_att: self.offscreen_color_att.release()
        if self.offscreen_depth_att: self.offscreen_depth_att.release()
        if self.offscreen_fbo and self.headless: # Only release if it's not ctx.screen and created for headless
            if self.ctx and self.offscreen_fbo != self.ctx.screen : # double check it's not screen
                 self.offscreen_fbo.release()
        if self.ctx: self.ctx.release()

        # Quit Pygame if it was initialized
        if not self.headless and self.display and pg.get_init():
            pg.quit()
        
        self.ctx = None # Mark as closed