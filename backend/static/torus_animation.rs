use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{Event, KeyCode, KeyEvent},
    execute,
    terminal::{size, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

fn clear_screen() -> Result<()> {
    execute!(std::io::stdout(), Clear(ClearType::All))
}

// ИСПРАВЛЕНО: Функция теперь принимает размеры экрана динамически.
fn draw_torus(rotations: f64, width: usize, height: usize) -> String {
    let mut output = vec![' '; width * height];
    let mut zbuffer = vec![0.0f64; width * height];

    let a = rotations; // Вращение вокруг оси X
    let b = rotations * 0.5; // Вращение вокруг оси Z

    let r1 = 1.0; // Радиус трубы
    let r2 = 2.0; // Радиус от центра до трубы

    // ИСПРАВЛЕНО: Увеличено расстояние (K2) с 5.0 до 15.0, чтобы избежать обрезания (clipping)
    // при вращении, когда часть тора подходит слишком близко к "камере".
    let k2 = 15.0;

    // ИСПРАВЛЕНО: Расчет масштаба (K1) теперь учитывает и высоту терминала.
    // Символ в консоли обычно в 2 раза выше, чем шире, поэтому умножаем height на 2.
    // Берем минимум, чтобы пончик вписывался и по ширине, и по высоте.
    // Коэффициент уменьшен с 3.0 до 2.0, чтобы оставить небольшие отступы.
    let screen_size = (width as f64).min(height as f64 * 2.0);
    let k1 = screen_size * k2 * 2.0 / (8.0 * (r1 + r2));

    let mut theta = 0.0f64;
    while theta < 2.0 * std::f64::consts::PI {
        let mut phi = 0.0f64;
        while phi < 2.0 * std::f64::consts::PI {
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();

            let cos_a = a.cos();
            let sin_a = a.sin();
            let cos_b = b.cos();
            let sin_b = b.sin();

            // Координаты точки на поверхности тора до вращения
            let circle_x = r2 + r1 * cos_theta;
            let circle_y = r1 * sin_theta;

            // 3D координаты после вращения
            let x =
                circle_x * (cos_b * cos_phi + sin_a * sin_b * sin_phi) - circle_y * cos_a * sin_b;
            let y =
                circle_x * (sin_b * cos_phi - sin_a * cos_b * sin_phi) + circle_y * cos_a * cos_b;
            let z = k2 + cos_a * circle_x * sin_phi + circle_y * sin_a;
            let ooz = 1.0 / z; // One over Z

            // Проекция на 2D экран
            // ИСПРАВЛЕНО: Добавлен множитель 1.5 по оси X, чтобы компенсировать
            // пропорции символов и сделать пончик визуально шире.
            let xp = (width as f64 / 2.0 + k1 * 1.5 * ooz * x) as i32;
            let yp = (height as f64 / 2.0 - k1 * ooz * y) as i32; // Минус, так как Y в консоли идет вниз

            // Расчет освещенности (Luminance)
            let l = cos_phi * cos_theta * sin_b - cos_a * cos_theta * sin_phi - sin_a * sin_theta
                + cos_b * (cos_a * sin_theta - cos_theta * sin_a * sin_phi);

            if l > 0.0 {
                if xp >= 0 && xp < width as i32 && yp >= 0 && yp < height as i32 {
                    let idx = (xp + yp * width as i32) as usize;
                    if ooz > zbuffer[idx] {
                        zbuffer[idx] = ooz;
                        let luminance_index = (l * 8.0) as usize;
                        // ИСПРАВЛЕНО: Правильный набор символов для отображения освещенности
                        let chars = ".,-~:;=!*#$@";
                        output[idx] = chars.chars().nth(luminance_index.min(11)).unwrap_or('@');
                    }
                }
            }

            phi += 0.02;
        }
        theta += 0.07;
    }

    let mut buffer = String::with_capacity(width * height + height);
    for row in 0..height {
        for col in 0..width {
            buffer.push(output[row * width + col]);
        }
        // Не добавляем перевод строки после последней строки, чтобы избежать скроллинга
        if row < height - 1 {
            buffer.push('\n');
        }
    }
    buffer
}

fn main() -> Result<()> {
    execute!(std::io::stdout(), EnterAlternateScreen, Hide)?;
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    thread::spawn(move || {
        // COMMENT: ERROR? Original code had type mismatch (f64 vs Instant). Fixed to use Instant for time tracking.
        let start_time = Instant::now();

        while running_clone.load(Ordering::Relaxed) {
            let rotations = start_time.elapsed().as_secs_f64();

            // ИСПРАВЛЕНО: Получаем реальный размер терминала.
            // Если не удается получить, используем дефолтный 80x24.
            let (cols, rows) = size().unwrap_or((125, 24));
            let width = cols as usize;
            let height = rows as usize;

            // ИСПРАВЛЕНО: Вместо очистки экрана (ClearType::All), которая вызывает мерцание,
            // мы просто перемещаем курсор в начало (0, 0) и перезаписываем кадр.
            // Это стандартная техника для консольной анимации (double buffering emulation).
            execute!(std::io::stdout(), MoveTo(0, 0)).unwrap();

            let animation = draw_torus(rotations, width, height);
            // Используем print! вместо println!, чтобы избежать лишнего перевода строки в конце,
            // который может вызвать скроллинг экрана.
            print!("{}", animation);

            thread::sleep(Duration::from_millis(50));
        }
    });

    // COMMENT: ERROR? Original code used async EventStream in sync main. Replaced with sync poll/read.
    loop {
        if crossterm::event::poll(Duration::from_millis(100))? {
            if let Event::Key(KeyEvent {
                code: KeyCode::Esc, ..
            }) = crossterm::event::read()?
            {
                running.store(false, Ordering::Relaxed);
                break;
            }
        }
        // Check if thread stopped (though currently it only stops if we set running to false)
        if !running.load(Ordering::Relaxed) {
            break;
        }
    }

    execute!(std::io::stdout(), Show, LeaveAlternateScreen)?;
    Ok(())
}
