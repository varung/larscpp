addpath /afs/cs.stanford.edu/u/silee/research/models/l1logreg/IRLS_LARS_new/lars

ntests = 1000000;
nsamples = 100;
nfeatures = 100;

for i = 1 : ntests
   fprintf('%g/%d\n',i,ntests);
   x = randn(nsamples, nfeatures);
   y = randn(nsamples, 1);
   w = x\y;
   C = 0.5*norm(w,1);
   theta = lars(x, y, 'lasso', C)';
   betapp = larspp(x, y, 'lasso', 'norm', C);thetapp = betapp(:,end);
   disc = max( abs( theta-thetapp ) );

   if(disc>1.0e-10)
      fprintf('%g	lars:%g (%g)	larspp:%g (%g)	C:%g\n',disc,norm(theta,1),full(sum(sign(abs(theta)))),norm(thetapp,1),full(sum(sign(abs(thetapp)))), C);
      obj_lars = sum( sum( (x*theta-y).^2 ) );
      obj_larspp = sum( sum( (x*thetapp-y).^2 ) );
      fprintf('		lars:%g	larspp:%g\n',obj_lars, obj_larspp );
      if( obj_lars<obj_larspp && abs(norm(theta,1)-C)<1.0e-10 )
         fprintf('PROBLEM!\n');
         pause;
      end
   end
end

